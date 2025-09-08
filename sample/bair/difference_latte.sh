#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Folder name to save the generated videos.
folder_name=bair_difflatte_ddim50_tweedie_keepnoise_1.0

echo torchrun --nnodes=1 --nproc_per_node=1 sample/sample_difference_ddp.py \
--config ./configs/bair/difflatte/bair_difflatte_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/bair64/008-DiffLatte-B-2-F16S1-bair64/checkpoints/1120000.pt \
--save_video_path ./test/bair/${folder_name} \
--tweedie-threshold 1.0

torchrun --nnodes=1 --nproc_per_node=1 sample/sample_difference_ddp.py \
--config ./configs/bair/difflatte/bair_difflatte_sample.yaml \
--ckpt  /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/bair64/008-DiffLatte-B-2-F16S1-bair64/checkpoints/1120000.pt \
--save_video_path ./test/bair/${folder_name} \
--tweedie-threshold 1.0

echo python tools/convert_videos_to_frames.py -s ./test/bair/${folder_name} -t ./test/frames/bair/${folder_name} --video_ext mp4 --target_size 64

python tools/convert_videos_to_frames.py -s ./test/bair/${folder_name} -t ./test/frames/bair/${folder_name} --video_ext mp4 --target_size 64

echo python tools/calc_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/train \
--fake_data_path ./test/frames/bair/${folder_name} \
--mirror 1 --gpus 1 --resolution 64 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0

python tools/calc_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/train \
--fake_data_path ./test/frames/bair/${folder_name} \
--mirror 1 --gpus 1 --resolution 64 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0