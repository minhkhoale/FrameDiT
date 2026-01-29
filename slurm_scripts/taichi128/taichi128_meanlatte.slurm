#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_difflattev1-noise-difference
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128_difflattev1-noise-difference-%j.out
#SBATCH --error=slurm_log/taichi128_difflattev1-noise-difference-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_mean.py --config ./configs/taichi128/meanlatte/taichi128_meanlatte_train.yaml "$@"