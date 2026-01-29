#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=taichi_fusedmatlatte-128-512_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi_fusedmatlatte-128-512_train-%j.out 
#SBATCH --error=slurm_log/taichi_fusedmatlatte-128-512_train-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi/fusedmatlatte/taichi_fusedmatlatte-128-512_train.yaml "$@"