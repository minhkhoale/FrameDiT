#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=get-videoufo-latents
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --mem=128G
#SBATCH --constraint=gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/dataset/%x-%j.out 
#SBATCH --error=slurm_log/dataset/%x-%j.err

source ~/.bashrc

conda activate tn

srun python scripts/videoufo/ddp_process_latents_v1.py \
    --csv_path /weka/s224075134/temporal_diffusion/datasets/video/videoufo/VideoUFO.csv \
    --model_path maxin-cn/Latte-1 \
    --output_path /weka/s224075134/temporal_diffusion/datasets/video/videoufo/autoencoder_kl_latents \
    --video_path /weka/s224075134/temporal_diffusion/datasets/video/videoufo/raw/VideoUFO_tar/VideoUFO \
    --num_workers 2 --batch-size 1