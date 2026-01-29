#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=MatLatteIMG-taichi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h200
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/MatLatteIMG-taichi-%j.out 
#SBATCH --error=slurm_log/MatLatteIMG-taichi-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/taichi/matlatteimg/taichi_matlatteimg-128-2048_train.yaml "$@"