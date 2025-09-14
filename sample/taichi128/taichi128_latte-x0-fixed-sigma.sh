#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi128/latte/taichi128_latte-x0-fixed-sigma_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/latte_based_vdm/results/taichi128/024-Latte-M-2-F16S3-taichi128/checkpoints/0580000.pt \
--save_video_path ./generation/taichi128/taichi128_latte-x0-fixed-sigma_580k_ddpm_250 \
