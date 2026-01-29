#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_difference_ddp.py \
--config ./configs/taichi128/difflatte/taichi128_difflatte_sample.yaml \
--ckpt /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/taichi128/008-DiffLatte-M-2-F16S3-taichi128/checkpoints/1000000.pt \
--tweedie-threshold 1.0 \
--save_video_path ./generation/debug/taichi128/taichi128_difflatte_1000k_ddpm_250_tweedie-1.0


# python tools/my_cal_metrics_for_dataset.py \
# --real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi128_reconstruction/train \
# --fake_data_path ./generation/taichi128/taichi128_difflatte_780k_ddpm_250_tweedie-0.5 \
# --mirror --resolution 128 \
# --verbose
