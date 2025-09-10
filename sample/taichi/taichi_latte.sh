#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi/latte/taichi_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/taichi256/002-Latte-XL-2-F16S3-taichi256/checkpoints/0560000.pt \
--save_video_path ./generation/debug/taichi/taichi_latte_560k_ddpm_250 \
