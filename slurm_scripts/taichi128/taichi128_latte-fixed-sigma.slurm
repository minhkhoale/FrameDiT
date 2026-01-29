#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128-Latte-Fixed-Sigma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-v100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128-Latte-Fixed-Sigma-%j.out 
#SBATCH --error=slurm_log/Taichi128-Latte-Fixed-Sigma-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/latte/taichi128_latte-fixed-sigma_train.yaml "$@"