#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python sample/sample_ddp.py \
--config ./configs/taichi/latte/taichi_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/Latte/results/027-Latte-XL-2-F16S3-taichi/checkpoints/0360000.pt \
--save_video_path ./test/taichi_latte
