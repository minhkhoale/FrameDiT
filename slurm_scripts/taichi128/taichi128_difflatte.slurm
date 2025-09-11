#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=DiffLatte-taichi128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-l40s
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference.py --config ./configs/taichi128/difflatte/taichi128_difflatte_train.yaml "$@"