#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_difflattev1-noise-correlation
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128_difflattev1-noise-correlation-%j.out
#SBATCH --error=slurm_log/taichi128_difflattev1-noise-correlation-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference_v1.py --config ./configs/taichi128/difflatte_v1/taichi128_difflattev1-noise-correlation_train.yaml "$@"