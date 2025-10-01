#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128_MatLatte-64-64_identity-row-bias
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128_MatLatte-64-64_identity-row-bias-%j.out 
#SBATCH --error=slurm_log/Taichi128_MatLatte-64-64_identity-row-bias-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/matlatte/taichi128_matlatte-64-64-identity-row-bias_train.yaml "$@"