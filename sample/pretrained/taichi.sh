#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi/latte/taichi_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/Latte/taichi-hd.pt \
--save_video_path ./test/pretrained/taichi_latte
