#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128-128_FusedMatLatte-64-256-concat-softmax
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=h100-m-01
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128-128_FusedMatLatte-64-256-concat-softmax-%j.out
#SBATCH --error=slurm_log/Taichi128-128_FusedMatLatte-64-256-concat-softmax-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/fusedmatlatte/taichi128-128_fusedmatlatte-m-64-256-concat-softmax_train.yaml "$@"