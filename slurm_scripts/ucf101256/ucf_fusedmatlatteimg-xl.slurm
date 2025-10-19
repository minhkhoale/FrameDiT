#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=UCF-fusedmatlatteimg-xl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/UCF-fusedmatlatteimg-xl-%j.out 
#SBATCH --error=slurm_log/UCF-fusedmatlatteimg-xl-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/ucf101/fusedmatlatteimg/ucf101_fusedmatlatteimg-xl_train.yaml "$@"