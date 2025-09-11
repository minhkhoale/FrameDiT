#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi128/matlatte/taichi128_matlatte-64-64_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/latte_based_vdm/results/taichi128/011-MatLatte-M-64-64-2-F16S3-taichi128/checkpoints/0510000.pt \
--save_video_path ./generation/taichi128/taichi128_matlatte_510k_ddpm_250 \
