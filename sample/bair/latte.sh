#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python sample/my_sample.py \
--config ./configs/bair/latte_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/bair64/005-Latte-B-2-F16S1-bair64/checkpoints/0250000.pt \
--save_video_path ./test/bair_latte