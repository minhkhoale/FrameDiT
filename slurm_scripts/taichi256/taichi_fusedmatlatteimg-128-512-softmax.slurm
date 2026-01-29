#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=taichi_fusedmatlatteimg-128-512_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi_fusedmatlatteimg-128-512_train-%j.out 
#SBATCH --error=slurm_log/taichi_fusedmatlatteimg-128-512_train-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/taichi/fusedmatlatteimg/taichi_fusedmatlatteimg-xl-loadpixel-128-512-softmax_train.yaml "$@"