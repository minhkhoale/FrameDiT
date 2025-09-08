#!/usr/bin/env bash
#SBATCH --job-name=Latte-UCF101_256_16
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00 
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/ucf101/latte/ucf101_latte_train.yaml "$@"