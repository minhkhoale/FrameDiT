#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=MatLatteIMG-taichi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h200
#SBATCH --mem=128G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/MatLatteIMG-taichi-%j.out 
#SBATCH --error=slurm_log/MatLatteIMG-taichi-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/taichi/matlatteimg/taichi_matlatteimg-256-1024-xl-pixel_train.yaml "$@"