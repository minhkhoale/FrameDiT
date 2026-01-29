#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi/latte_img/taichi_latteimg_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results_img/taichi_img256/001-LatteIMG-XL-2-F16S3-taichi_img256/checkpoints/0520000.pt \
--save_video_path ./generation/debug/taichi_img/taichi_img_latteimg_520k_ddpm_250 \

python tools/my_cal_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi/train \
--fake_data_path ./generation/debug/taichi_img/taichi_img_latteimg_520k_ddpm_250 \
--mirror --resolution 256 \
--verbose


