#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# This script runs a multi-run benchmark and captures a deep profile on the first run.
# - `--num-runs`: Controls the number of benchmark iterations for stable, averaged metrics.
# - `--profile`: Enables cProfile and torch.profiler for a detailed trace of the first run.
# - `--profile_memory`: Enables memory tracking within the torch.profiler.
python -m scripts.inference_profiler \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 10 \
    --guidance_scale 2 \
    --video_path "assets/demo_video.mp4" \
    --audio_path "assets/demo_audio.wav" \
    --video_out_path "video_out.mp4" \
    --num_profile_runs 2 \
    --profile \
    --profile_memory

#     --enable_deepcache \