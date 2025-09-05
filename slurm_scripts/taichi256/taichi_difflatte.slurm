#!/usr/bin/env bash
#SBATCH --job-name=DiffLatte-Taichi16_256
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference.py --config ./configs/taichi/taichi_difflatte_train.yaml "$@"