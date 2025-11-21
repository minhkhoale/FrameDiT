#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=matlatte-64-256-softmax-taichi128-128
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --exclude=h100-m-01
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/matlatte-64-256-softmax-taichi128-128-%j.out 
#SBATCH --error=slurm_log/matlatte-64-256-softmax-taichi128-128-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/matlatte/taichi128-128-matlatte-64-256-softmax_train.yaml "$@"