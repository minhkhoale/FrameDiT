#!/usr/bin/env bash
#SBATCH --job-name=DiffLatteInDirect-Taichi256_16
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --mem=32G
#SBATCH --constraint=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/DiffLatteInDirect-Taichi256_16-%j.out 
#SBATCH --error=slurm_log/DiffLatteInDirect-Taichi256_16-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference_v2.py --config ./configs/taichi/difflatte/taichi_difflatte-x0-fixed-sigma_train.yaml "$@"