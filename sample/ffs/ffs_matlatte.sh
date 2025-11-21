#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/ffs/matlatte/ffs_matlatte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/ffs256/001-MatLatte-XL-256-256-2-F16S3-ffs256/checkpoints/0340000.pt \
--save_video_path ./generation/ffs/ffs_matlatte_340k_ddpm_250 \


torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/ucf101/matlatte/ucf101_matlatte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/ucf101256/004-MatLatte-XL-256-256-2-F16S3-ucf101256/checkpoints/0340000.pt \
--save_video_path ./generation/ucf/ucf_matlatte_340k_ddpm_250 \


python tools/my_cal_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video/faceforensics/train/videos \
--fake_data_path /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/generation/ffs/ffs_matlatte_340k_ddpm_250 \
--mirror --resolution 256 \
--verbose

