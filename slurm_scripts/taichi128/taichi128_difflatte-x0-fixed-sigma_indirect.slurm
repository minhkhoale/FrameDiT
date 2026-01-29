#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_taichi_difflatte-x0-fixed-sigma_indirect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-l40s
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128_taichi_difflatte-x0-fixed-sigma_indirect-%j.out 
#SBATCH --error=slurm_log/taichi128taichi_difflatte-x0-fixed-sigma_indirect-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference_v2.py --config ./configs/taichi128/difflatte_indirect/taichi_difflatte-x0-fixed-sigma_indirect_train.yaml "$@"