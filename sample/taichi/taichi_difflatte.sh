#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_difference_ddp.py \
--config ./configs/taichi/difflatte/taichi_difflatte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/taichi256/004-DiffLatte-XL-2-F16S3-taichi256/checkpoints/0190000.pt \
--tweedie-threshold 1.0 \
--save_video_path ./generation/taichi/taichi_difflatte_190k_ddpm_250_tweedie-1.0 \
