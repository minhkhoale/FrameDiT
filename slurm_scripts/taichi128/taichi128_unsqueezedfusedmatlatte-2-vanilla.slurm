#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_unsqueezedfusedmatlatte-onlylocal_train
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128_unsqueezedfusedmatlatte-onlylocal_train-%j.out
#SBATCH --error=slurm_log/taichi128_unsqueezedfusedmatlatte-onlylocal_train-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/unsqueezedfusedmatlatte/taichi128_unsqueezedfusedmatlatte-vanilla_train.yaml "$@"