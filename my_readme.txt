# Preprocess data to compute FVD

## BAIR

### Real
python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/datasets/video/bair/softmotion30_44k/test/video_aux1 -t /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/test --video_ext mp4 --target_size 64

python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/datasets/video/bair/softmotion30_44k/train/video_aux1 -t /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/train --video_ext mp4 --target_size 64

### Fake
python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/bair_latte -t /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/frames/bair_latte --video_ext mp4 --target_size 64

python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/bair_difflatte -t /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/frames/bair_difflatte --video_ext mp4 --target_size 64

python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/bair_difflatte_tweedie0.5 -t /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/frames/bair_difflatte_tweedie0.5 --video_ext mp4 --target_size 64


## TAICHI
python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/datasets/video/taichi/test -t /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/taichi/test --video_ext mp4 --target_size 256


# Convert mp4 to jpg
## TAICHI
python tools/convert_videos_to_frames.py -s /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/pretrained/taichi_latte -t /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/pretrained/frames/taichi_latte --video_ext mp4 

# FVD

## TAICHI
python tools/calc_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi/train \
--fake_data_path /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/pretrained/frames/taichi_latte \
--mirror 1 --gpus 1 --resolution 256 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0

## BAIR
python tools/calc_metrics_for_dataset.py \
--real_data_path /scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/bair/train \
--fake_data_path /scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/test/frames/bair_difflatte_tweedie0.5 \
--mirror 1 --gpus 1 --resolution 64 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0