#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ffs_fusedmatlatteimg_128_1024-concat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/ffs_fusedmatlatteimg_128_1024-concat-%j.out 
#SBATCH --error=slurm_log/ffs_fusedmatlatteimg_128_1024-concat-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/ffs/fusedmatlatteimg/ffs_fusedmatlatteimg-128-1024-concat_train.yaml "$@"