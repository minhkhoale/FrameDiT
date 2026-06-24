#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101128-matlatte-gamma-1.0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/ucf101128_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/ucf101128_adaptive_schedule/%x-%j.out

source ~/.bashrc

conda activate latte

srun python my_train_accelerate.py --config configs/ucf101_adaptive_schedule/ucf101_128_matlatte_gamma_1.0_train.yaml
