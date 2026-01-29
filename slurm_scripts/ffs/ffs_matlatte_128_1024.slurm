#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=FFS-MatLatte-128-1024
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/FFS-MatLatte-128-1024-%j.out 
#SBATCH --error=slurm_log/FFS-MatLatte-128-1024-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/ffs/matlatte/ffs_matlatte-128-1024_train.yaml "$@"