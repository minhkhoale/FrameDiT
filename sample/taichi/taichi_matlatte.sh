#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi/matlatte/taichi_matlatte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/taichi256/003-MatLatte-XL-256-256-2-F16S3-taichi256/checkpoints/0440000.pt \
--save_video_path ./generation/taichi/taichi_matlatte_440k_ddpm_250


python tools/my_cal_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi/train \
--fake_data_path ./generation/taichi/taichi_matlatte_440k_ddpm_250 \
--mirror --resolution 256 \
--verbose
