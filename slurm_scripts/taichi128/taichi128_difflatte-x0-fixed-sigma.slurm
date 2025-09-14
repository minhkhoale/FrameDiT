#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_difflatte-x0-fixed-sigma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-l40s
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/taichi128_difflatte-x0-fixed-sigma-%j.out 
#SBATCH --error=slurm_log/taichi128_difflatte-x0-fixed-sigma-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference.py --config ./configs/taichi128/difflatte/taichi128_difflatte-x0-fixed-sigma_train.yaml "$@"