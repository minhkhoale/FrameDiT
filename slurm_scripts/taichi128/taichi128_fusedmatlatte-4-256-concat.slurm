#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=Taichi128_FusedMatLatte-4-256-concat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128_FusedMatLatte-4-256-concat-%j.out
#SBATCH --error=slurm_log/Taichi128_FusedMatLatte-4-256-concat-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/fusedmatlatte/taichi128_fusedmatlatte-4-256-concat_train.yaml "$@"