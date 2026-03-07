#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=get-pexels-latents
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%x-%j.out 
#SBATCH --error=slurm_log/%x-%j.err

source ~/.bashrc

conda activate tn

srun python scripts/pexels/ddp_process_latents_v1.py \
--parquet_path /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels_400k.parquet \
--model_path maxin-cn/Latte-1 \
--output_path /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k_v1/pexels400k_processed_t2v \
--video_path /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k_v1/videos \
--num_workers 2 --batch-size 2