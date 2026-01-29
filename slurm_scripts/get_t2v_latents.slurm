#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --constraint=gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%x-%j.out 
#SBATCH --error=slurm_log/%x-%j.err

source ~/.bashrc

conda activate tn

python scripts/get_t2v_latents.py \
 --csv_path /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD.csv \
  --save_dir /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/latents \
  --n_videos 200000 \
  --repo_id nkp37/OpenVid-1M \
  --zip_contains HD \
  --batch_size 4 \
  --num_workers 4 \
  --video_length 16 \
  --target_h 512 --target_w 512 \
  --latent_scale 1.0 \
  --save_layout T_C \
  --zip_index_json /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD.json