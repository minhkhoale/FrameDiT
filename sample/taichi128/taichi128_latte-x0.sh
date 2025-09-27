#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi128/latte/taichi128_latte-x0_sample.yaml \
--ckpt ./results/taichi128/001-Latte-M-2-F16S3-taichi128/checkpoints/0510000.pt \
--save_video_path ./generation/taichi128/taichi128_latte-x0_510k_ddpm_250 \

python tools/my_cal_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi128_reconstruction/train \
--fake_data_path ./generation/taichi128/taichi128_latte-x0_510k_ddpm_250 \
--mirror --resolution 128 \
--verbose
