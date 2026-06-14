#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=UCF-fusedmatlatteimg-xl-pixel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/UCF/%x-%j.out 
#SBATCH --error=slurm_log/UCF/%x-%j.err

source ~/.bashrc

conda activate tn

srun python train_with_img.py --config ./configs/ucf101/fusedmatlatteimg/ucf101_fusedmatlatteimg-xl-pixel_train.yaml "$@"