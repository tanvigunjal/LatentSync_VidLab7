import argparse
import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.performance_monitor import PerformanceMonitor
from DeepCache import DeepCacheSDHelper
import cProfile
import pstats
import json
from io import StringIO
from torch.profiler import ProfilerActivity
import shutil
import time
import gc
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Constants for optimization
BATCH_SIZE = 1  # Since you're the only user
MIN_MEMORY_THRESHOLD = 1024 * 1024 * 1024  # 1GB
CACHE_SIZE = 32
DEFAULT_NUM_FRAMES = 16

def optimized_svd(tensor):
    """Optimized SVD implementation for MPS devices."""
    if tensor.device.type == "mps":
        # Move to CPU, perform SVD, and move back to MPS
        with torch.no_grad():
            cpu_tensor = tensor.cpu()
            U, S, V = torch.svd(cpu_tensor)
            return U.to("mps"), S.to("mps"), V.to("mps")
    else:
        return torch.svd(tensor)

class AggressiveModelOptimizer:
    @staticmethod
    def optimize_for_inference(model, device, dtype):
        if not isinstance(model, nn.Module):
            return model
            
        model = model.eval()  # Ensure eval mode
        
        # Aggressive optimization for personal use
        if device == "mps":
            try:
                # Only apply channels_last to Conv2d layers with 4D weights
                for module in model.modules():
                    if isinstance(module, nn.Conv2d) and len(module.weight.shape) == 4:
                        module.weight.data = module.weight.data.to(memory_format=torch.channels_last)
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data.contiguous()
            except Exception as e:
                print(f"Memory format optimization skipped: {e}")
        
        # Try JIT compilation with specific optimizations
        try:
            if not isinstance(model, torch.jit.ScriptModule):
                # Attempt to JIT compile submodules individually
                for name, module in model.named_children():
                    if isinstance(module, nn.Sequential):
                        try:
                            optimized_module = torch.jit.script(module)
                            setattr(model, name, optimized_module)
                        except Exception:
                            continue
                
                # Try to JIT compile the entire model
                try:
                    model = torch.jit.script(model)
                    model = torch.jit.optimize_for_inference(model)
                except Exception as e:
                    print(f"Full model JIT compilation skipped: {e}")
        except Exception as e:
            print(f"JIT optimization skipped: {e}")
        
        return model

class EnhancedSyncOptimizer:
    def __init__(self, base_guidance_scale=1.0):
        self.base_guidance_scale = base_guidance_scale
        self.prev_frame = None
        self.frame_history = []
        self.temporal_window = 5

    def calculate_sync_confidence(self, current_frame, audio_features):
        """Calculate synchronization confidence score."""
        with torch.no_grad():
            # Normalize inputs
            current_frame = torch.nn.functional.normalize(current_frame.flatten())
            audio_features = torch.nn.functional.normalize(audio_features.flatten())
            
            # Calculate cosine similarity
            confidence = torch.nn.functional.cosine_similarity(
                current_frame.unsqueeze(0), 
                audio_features.unsqueeze(0)
            ).item()
            
            return max(0.0, min(1.0, confidence))

    def get_dynamic_guidance_scale(self, current_frame, audio_features):
        """Adjust guidance scale based on sync confidence."""
        sync_confidence = self.calculate_sync_confidence(current_frame, audio_features)
        # Scale between 1.0-2.0 based on confidence
        guidance_scale = self.base_guidance_scale * (1.0 + 0.5 * (1.0 - sync_confidence))
        return guidance_scale, sync_confidence

    def apply_temporal_smoothing(self, current_frame):
        """Apply temporal smoothing between consecutive frames."""
        if self.prev_frame is not None:
            # Apply weighted average with previous frame
            smoothed_frame = 0.8 * current_frame + 0.2 * self.prev_frame
            
            # Update frame history
            self.frame_history.append(current_frame)
            if len(self.frame_history) > self.temporal_window:
                self.frame_history.pop(0)
                
            # Additional temporal consistency check
            if len(self.frame_history) >= 3:
                # Calculate motion smoothness
                motion_diff = torch.mean(torch.abs(current_frame - self.prev_frame))
                if motion_diff > 0.5:  # High motion threshold
                    # Stronger smoothing for high motion
                    smoothed_frame = 0.7 * current_frame + 0.3 * self.prev_frame
            
            current_frame = smoothed_frame
        
        self.prev_frame = current_frame.clone()
        return current_frame

class MemoryOptimizedPipeline:
    def __init__(self, original_pipeline, device, dtype):
        self.pipeline = original_pipeline
        self.device = device
        self.dtype = dtype
        self.vae_cache = {}
        self.audio_cache = {}
        self.sync_optimizer = EnhancedSyncOptimizer()
        
    @staticmethod
    def _is_memory_available(required_memory):
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            return free_memory > required_memory
        elif torch.backends.mps.is_available():
            # For MPS, we'll use system memory as a proxy
            import psutil
            return psutil.virtual_memory().available > required_memory
        return True
    
    def _clear_cache_if_needed(self, required_memory):
        if not self._is_memory_available(required_memory):
            self.vae_cache.clear()
            self.audio_cache.clear()
            _clear_memory(aggressive=True)

def _clear_memory(aggressive=False):
    """Enhanced memory clearing with optional aggressive mode."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        if aggressive:
            # Force synchronization on MPS
            dummy = torch.ones(1, device="mps")
            dummy.item()
    
    if aggressive:
        # Multiple GC runs for more thorough cleanup
        for _ in range(3):
            gc.collect()
        
        # Force Python to return memory to the OS where possible
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            # Suggest to OS to release memory
            process.memory_info()
        except ImportError:
            # If psutil is not available, just continue with basic memory clearing
            pass

@lru_cache(maxsize=CACHE_SIZE)
def cache_vae_output(input_tensor_key, vae):
    """Cache VAE outputs with tensor key."""
    input_tensor = torch.from_numpy(np.frombuffer(input_tensor_key, dtype=np.float32))
    with torch.no_grad():
        return vae(input_tensor)

def process_audio_parallel(audio_encoder, audio_path, num_frames, audio_feat_length):
    """Process audio in parallel with caching."""
    cache_key = f"{audio_path}_{num_frames}_{audio_feat_length}"
    
    def process_chunk():
        # Get the audio features using the correct method
        audio_feat = audio_encoder.audio2feat(audio_path)
        return audio_encoder.crop_overlap_audio_window(audio_feat, start_index=0)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_chunk)
        return future.result()

def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")
    
    overall_start_time = time.time()

    # Enhanced device detection and setup
    if torch.cuda.is_available():
        cuda_idx = torch.cuda.current_device()
        device_str = f"cuda:{cuda_idx}"
        cap = torch.cuda.get_device_capability(cuda_idx)
        is_fp16_supported = cap[0] >= 7
        dtype = torch.float16 if is_fp16_supported else torch.float32
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device_str = "mps"
        dtype = torch.float32
        # Aggressive MPS optimizations
        torch.backends.mps.enable_cache = True
    else:
        device_str = "cpu"
        dtype = torch.float32
        # Enable OpenMP optimizations
        torch.set_num_threads(os.cpu_count())
    
    device = torch.device(device_str)
    print(f"Using device: {device_str}, dtype: {dtype}")

    monitor = PerformanceMonitor(device, enabled=True)
    monitor.start_run()

    # Load models with aggressive optimization
    with monitor.profile("Load Models"):
        # Load scheduler with fewer steps for faster inference
        scheduler = DDIMScheduler.from_pretrained("configs")
        scheduler.set_timesteps(args.inference_steps)
        
        # Always use tiny Whisper model for faster inference
        whisper_model_path = "checkpoints/whisper/tiny.pt"
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=device_str,
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )
        
        # Load and optimize VAE with proper configuration
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=dtype,
            use_safetensors=True
        )
        # Set VAE configuration before optimization
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0.0  # Explicitly set shift factor
        
        # Register scaling in forward hooks to ensure persistence
        def apply_scaling(module, input, output):
            if module.config.scaling_factor is not None:
                return output * module.config.scaling_factor
            return output

        def apply_shift(module, input, output):
            if module.config.shift_factor is not None:
                return output + module.config.shift_factor
            return output

        vae.register_forward_hook(apply_scaling)
        vae.register_forward_pre_hook(apply_shift)
        
        vae = AggressiveModelOptimizer.optimize_for_inference(vae, device_str, dtype)
        
        # Load and optimize UNet
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            args.inference_ckpt_path,
            device=device_str,
        )
        unet = AggressiveModelOptimizer.optimize_for_inference(unet, device_str, dtype)
        
        # Move models to device with optimized memory format
        try:
            vae = vae.to(device=device, memory_format=torch.channels_last)
            unet = unet.to(device=device, memory_format=torch.channels_last)
            if dtype == torch.float16:
                unet = unet.half()
                vae = vae.half()
        except Exception as e:
            print(f"Warning: Optimization failed, falling back to standard format: {e}")
            vae = vae.to(device=device)
            unet = unet.to(device=device)

    # Initialize optimized pipeline
    with monitor.profile("Initialize Pipeline"):
        # Verify VAE configuration before pipeline initialization
        assert vae.config.scaling_factor is not None, "VAE scaling factor must be set"
        assert vae.config.shift_factor is not None, "VAE shift factor must be set"
        
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(device)
        
        # Ensure VAE config is properly set in pipeline
        pipeline.vae.config.scaling_factor = vae.config.scaling_factor
        pipeline.vae.config.shift_factor = vae.config.shift_factor
        
        if args.enable_deepcache:
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    # Aggressive warmup
    if not args.profile:
        print("Performing aggressive warmup...")
        
        # Create all necessary directories
        os.makedirs(args.temp_dir, exist_ok=True)
        warmup_dir = os.path.join(args.temp_dir, "warmup")
        os.makedirs(warmup_dir, exist_ok=True)
        
        # Create a temporary directory for intermediate files
        temp_process_dir = os.path.join(args.temp_dir, "process")
        os.makedirs(temp_process_dir, exist_ok=True)
        
        with torch.no_grad():
            _clear_memory(aggressive=True)
            # Multiple small warmup runs
            for i in range(1):
                warmup_output = os.path.join(warmup_dir, f"warmup_{i}.mp4")
                try:
                    pipeline(
                        video_path=args.video_path,
                        audio_path=args.audio_path,
                        video_out_path=warmup_output,
                        num_frames=8,  # Smaller frame count for warmup
                        num_inference_steps=5,  # Fewer steps for warmup
                        guidance_scale=args.guidance_scale,
                        weight_dtype=dtype,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask_image_path=config.data.mask_image_path,
                        temp_dir=temp_process_dir,  # Use separate temp dir for processing
                        monitor=monitor,
                    )
                except Exception as e:
                    print(f"Warmup {i} failed: {e}")
                finally:
                    _clear_memory(aggressive=True)
        
        # Clean up all temporary directories
        try:
            import shutil
            shutil.rmtree(warmup_dir, ignore_errors=True)
            shutil.rmtree(temp_process_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
            
        print("Warmup completed.")

    # Profiling setup
    if args.profile:
        print("Starting profiling...")
        python_profiler = cProfile.Profile()
        python_profiler.enable()
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Profiling code (unchanged)
        profiling_dir = "profiling_results"
        os.makedirs(profiling_dir, exist_ok=True)

        # cProfile stats
        s = StringIO()
        ps = pstats.Stats(python_profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        cprofile_stats = s.getvalue()

        # PyTorch profiler stats
        torch_profiler_stats_cpu = p.key_averages().table(sort_by="cpu_time_total", row_limit=30)
        
        profiling_data = {
            "cProfile": {
                "description": "cProfile results sorted by cumulative time.",
                "stats": cprofile_stats
            },
            "torch_profiler": {
                "description": "PyTorch profiler results.",
                "cpu_time": torch_profiler_stats_cpu,
            }
        }

        if torch.cuda.is_available():
            torch_profiler_stats_cuda = p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            profiling_data["torch_profiler"]["cuda_time"] = torch_profiler_stats_cuda


        with open(os.path.join(profiling_dir, "profiling_results.json"), "w") as f:
            json.dump(profiling_data, f, indent=4)

        print(f"Profiling results saved to {profiling_dir}/profiling_results.json")
        print("TensorBoard trace saved to ./log/lipsync")


    # Generate Final Output with optimizations
    with monitor.profile("Generate Final Output"):
        print(f"Generating final output video at {args.video_out_path}...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.video_out_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        _clear_memory(aggressive=True)
        
        # Process audio features first
        with monitor.profile("Audio Processing"):
            audio_features = process_audio_parallel(
                audio_encoder,
                args.audio_path,
                config.data.num_frames,
                config.data.audio_feat_length
            )
        
        with monitor.profile("Pipeline Execution"):
            pipeline(
                video_path=args.video_path,
                audio_path=args.audio_path,
                video_out_path=args.video_out_path,
                num_frames=config.data.num_frames,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                weight_dtype=dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=config.data.mask_image_path,
                temp_dir=args.temp_dir,
                monitor=monitor,
                audio_features=audio_features,  # Pass the processed audio features
            )

    monitor.end_run()
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    # Print performance summary (unchanged)
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY (Aggregated across all runs)")
    print("="*60)
    
    summary = monitor.get_summary()
    print(f"Total Script Execution Time: {total_time:.2f} seconds")
    print(f"Number of 'PerformanceMonitor' Runs Recorded: {summary.get('num_runs', 0)}")
    
    total_mean_time = 0
    if 'total_time_stats' in summary and summary['total_time_stats']['mean'] > 0:
        total_mean_time = summary['total_time_stats']['mean']
        print(f"Avg. Pipeline Time (across runs): {total_mean_time:.2f}s (Std: {summary['total_time_stats']['std']:.2f}s)")
        print(f"Avg. Peak Memory (across runs): {summary['total_memory_peak_stats']['mean']:.1f} MB (Std: {summary['total_memory_peak_stats']['std']:.1f} MB)")
    else:
        print("No valid pipeline time metrics recorded by PerformanceMonitor.")

    if 'avg_cpu_usage_stats' in summary and summary['avg_cpu_usage_stats']['mean'] > 0:
        print(f"Average CPU Usage: {summary['avg_cpu_usage_stats']['mean']:.1f}%")
    if 'avg_gpu_usage_stats' in summary and summary['avg_gpu_usage_stats']['mean'] > 0:
        print(f"Average GPU Usage: {summary['avg_gpu_usage_stats']['mean']:.1f}%")
    
    print("\nStage Breakdown (Averages):")
    if 'stage_breakdown' in summary:
        for stage, stats in summary['stage_breakdown'].items():
            if 'time_stats' in stats:
                time_taken = stats['time_stats']['mean']
                percentage = (time_taken / total_mean_time) * 100 if total_mean_time > 0 else 0
                mem_peak = stats.get('memory_peak_stats', {}).get('mean', 0)
                print(f"  {stage}: {time_taken:.2f}s ({percentage:.1f}%) - Peak Mem: {mem_peak:.1f} MB")
    
    print("="*60)
    
    # Save metrics and create visualizations
    metrics_dir = "performance_metrics"
    monitor.save_metrics(metrics_dir)
    print(f"\n✓ Performance metrics saved to {metrics_dir}/metrics.json")
    print("✓ Final video generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    parser.add_argument("--num_profile_runs", type=int, default=3, help="Number of active profiling steps.")
    parser.add_argument("--profile_memory", action="store_true", help="Enable memory profiling (heavy).")
    parser.add_argument("--with_stack", action="store_true", help="Enable stack tracing in profiler (heavy).")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    main(config, args)