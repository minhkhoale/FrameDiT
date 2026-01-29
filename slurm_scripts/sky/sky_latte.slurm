#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=Latte-taichi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/sky/latte/sky_latte_train.yaml "$@"