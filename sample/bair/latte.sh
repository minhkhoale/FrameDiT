#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

folder_name=bair_latte_ddim50

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_ddp.py \
--config ./configs/bair/latte/bair_latte_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/bair64/007-Latte-B-2-F16S1-bair64/checkpoints/1270000.pt \
--save_video_path ./test/bair/${folder_name}

python tools/convert_videos_to_frames.py -s ./test/bair/${folder_name} -t ./test/frames/bair/${folder_name} --video_ext mp4 --target_size 64

python tools/calc_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/train \
--fake_data_path ./test/frames/bair/${folder_name} \
--mirror 1 --gpus 1 --resolution 64 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0