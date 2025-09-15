#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128_MatLatte-64-256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-v100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128_MatLatte-64-256-%j.out 
#SBATCH --error=slurm_log/Taichi128_MatLatte-64-256-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/matlatte/taichi128_matlatte-64-256_train.yaml "$@"