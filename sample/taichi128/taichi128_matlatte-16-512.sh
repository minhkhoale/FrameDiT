#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/taichi128/matlatte/taichi128_matlatte-16-512_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/latte_based_vdm/results/taichi128/016-MatLatte-M-16-512-2-F16S3-taichi128/checkpoints/0800000.pt \
--save_video_path ./generation/taichi128/taichi128_matlatte-16-512_800k_ddpm_250 \


python tools/my_cal_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi128_reconstruction/train \
--fake_data_path ./generation/taichi128/taichi128_matlatte-16-512_800k_ddpm_250 \
--mirror --resolution 128 \
--verbose
