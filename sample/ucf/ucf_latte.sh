#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/ucf101/latte/ucf101_latte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/pretrained/ucf101.pt \
--save_video_path ./generation/ucf101_img256/pretrained_latte_class_5 \
