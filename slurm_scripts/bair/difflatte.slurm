#!/usr/bin/env bash
#SBATCH --job-name=DiffLatte-BAIR16
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference.py --config ./configs/bair/difflatte_train.yaml "$@"