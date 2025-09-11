#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi128/latte/taichi128_latte_sample.yaml \
--ckpt ./results/taichi128/013-Latte-M-2-F16S3-taichi128/checkpoints/0450000.pt \
--save_video_path ./generation/taichi128/taichi128_latte_450k_ddpm_250 \
