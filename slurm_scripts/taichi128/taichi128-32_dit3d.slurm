#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128-32_dit3d
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128-32_dit3d-%j.out 
#SBATCH --error=slurm_log/taichi128-32_dit3d-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/dit3d/taichi128-32_dit3d_train.yaml "$@"