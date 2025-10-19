#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128_FusedMatLatte-32-256-concat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Taichi128_FusedMatLatte-32-256-concat-%j.out
#SBATCH --error=slurm_log/Taichi128_FusedMatLatte-32-256-concat-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/fusedmatlatte/taichi128_fusedmatlatte-32-256-concat_train.yaml "$@"