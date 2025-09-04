#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python sample/sample.py \
--config ./configs/bair/latte_sample.yaml \
--ckpt  ./results/bair64/001-LatteV2-B-2-F16S1-bair64/checkpoints/0250000.pt \
--save_video_path ./test/bair
