#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=taichi_fusedmatlatte1d-256-1024-concat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%x-%j.out 
#SBATCH --error=slurm_log/%x-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi/fusedmatlatte1d/taichi_fusedmatlatte1d-256-1024_train.yaml "$@"