#!/usr/bin/env bash
#SBATCH --job-name=DiffLatte-Taichi256_16
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --constraint=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference.py --config ./configs/taichi/difflatte/taichi_difflatte_train.yaml "$@"