#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_difference_ddp.py \
--config ./configs/ucf101/difflatte/ucf101_difflatte_sample.yaml \
--ckpt ./results/ucf101256/001-DiffLatte-XL-2-F16S1-ucf101256/checkpoints/0460000.pt \
--save_video_path ./test/debug/ucf/pretrained_ucf_difflatte_ddpm_250 \
