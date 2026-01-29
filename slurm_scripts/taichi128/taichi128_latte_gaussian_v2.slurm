#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Latte-taichi128-gaussian_diffusion_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Latte-taichi128-gaussian_diffusion_v2-%j.out 
#SBATCH --error=slurm_log/Latte-taichi128-gaussian_diffusion_v2-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/latte/taichi128_latte_gaussianv2_train.yaml "$@"