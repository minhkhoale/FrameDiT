#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/ucf101/latte/ucf101_latte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/ucf101256/005-Latte-XL-2-F16S1-ucf101256/checkpoints/0410000.pt \
--save_video_path ./generation/debug/ucf/ucf_latte_560k_ddpm_250 \
