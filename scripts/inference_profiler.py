# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf
import torch
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

def _clear_memory():
    """Releases all possible memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")
    
    # Initialize performance monitor
    overall_start_time = time.time()

    # Detect available device: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        # use the current CUDA device index
        cuda_idx = torch.cuda.current_device()
        device_str = f"cuda:{cuda_idx}"
        # check for fp16 support (compute capability >= 7)
        try:
            cap = torch.cuda.get_device_capability(cuda_idx)
            is_fp16_supported = cap[0] >= 7
        except Exception:
            is_fp16_supported = False
        dtype = torch.float16 if is_fp16_supported else torch.float32
    elif getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        device_str = "mps"
        dtype = torch.float32
    else:
        device_str = "cpu"
        dtype = torch.float32  # Using float32 for better compatibility on CPU
    device = torch.device(device_str)

    print(f"Using device: {device_str}, dtype: {dtype}")

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")


    # This is a custom utility built by me to track performance metrics
    monitor = PerformanceMonitor(device, enabled=True)
    monitor.start_run()

    # Stage 1: Load scheduler
    with monitor.profile("Load Scheduler"):
        scheduler = DDIMScheduler.from_pretrained("configs")

    # Stage 2: Load audio encoder
    with monitor.profile("Load Audio Encoder"):
        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=device_str,
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

    # Stage 3: Load VAE
    with monitor.profile("Load VAE"):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

    # Stage 4: Load UNet
    with monitor.profile("Load UNet"):
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            args.inference_ckpt_path,
            device=device_str,
        )

        # move and cast model parameters
        try:
            unet = unet.to(device=device)
            if dtype == torch.float16:
                unet = unet.half()
        except Exception:
            # Fallback: try moving without dtype cast
            unet = unet.to(device=device)

    # Stage 5: Initialize pipeline
    with monitor.profile("Initialize Pipeline"):
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to(device)

        # use DeepCache
        if args.enable_deepcache:
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    print(f"Initial seed: {torch.initial_seed()}")

    if args.profile:
        # --- Profiling Setup ---
        # Ensure log directory exists
        os.makedirs("./log/lipsync", exist_ok=True)

        # Define profiler activities
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Define schedule based on new arg
        wait_steps = 1
        warmup_steps = 1
        active_steps = args.num_profile_runs
        num_torch_steps = wait_steps + warmup_steps + active_steps

        # Initialize profilers
        python_profiler = cProfile.Profile()
        torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait_steps,
                                             warmup=warmup_steps, 
                                             active=active_steps, 
                                             repeat=1
                                            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/lipsync'),
            record_shapes=True,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack
        )

        # --- Run Profiling ---
        # Use a temporary directory for intermediate profiling outputs to avoid I/O skew
        profiling_output_dir = "temp_profiling_output"
        os.makedirs(profiling_output_dir, exist_ok=True)
        temp_video_out_path = os.path.join(profiling_output_dir, "temp_video.mp4")

        print("Starting detailed profiling...")
        python_profiler.enable()
        with torch_profiler as p:
            for step in range(num_torch_steps): 
                _clear_memory() #memory clearing

                # Add printouts for clarity
                if step < wait_steps:
                    print(f"--- Profiler waiting... (step {step + 1}/{num_torch_steps}) ---")
                elif step < wait_steps + warmup_steps:
                    print(f"--- Profiler warming up... (step {step + 1}/{num_torch_steps}) ---")
                else:
                    print(f"--- Profiler ACTIVE... (step {step + 1}/{num_torch_steps}) ---")

                pipeline(
                    video_path=args.video_path,
                    audio_path=args.audio_path,
                    video_out_path=temp_video_out_path, # Write to temp path
                    num_frames=config.data.num_frames,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    weight_dtype=dtype,
                    width=config.data.resolution,
                    height=config.data.resolution,
                    mask_image_path=config.data.mask_image_path,
                    temp_dir=args.temp_dir,
                    monitor=monitor,
                )
                p.step()
        python_profiler.disable()
        print("Profiling finished.")

        # Clean up temporary profiling outputs
        shutil.rmtree(profiling_output_dir)

        # --- Save Profiling Results ---
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

    # --- Generate Final Output (always run) ---
    with monitor.profile("Generate Final Output"):
        print(f"Generating final output video at {args.video_out_path}...")
        _clear_memory()
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path, # Write to final path
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
            monitor=monitor,
        )
    print("Final video generated.")

    monitor.end_run()
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    # --- Print Summary ---
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