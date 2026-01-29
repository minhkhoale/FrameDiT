#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=taichi-Latteimg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi-Latteimg-%j.out 
#SBATCH --error=slurm_log/taichi-Latteimg-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/taichi/latte_img/taichi_latteimg_train.yaml "$@"