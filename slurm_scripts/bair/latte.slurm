#!/usr/bin/env bash
#SBATCH --job-name=Latte-BAIR16
#SBATCH --partition=gpu-large
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/bair/latte/bair_latte_train.yaml "$@"