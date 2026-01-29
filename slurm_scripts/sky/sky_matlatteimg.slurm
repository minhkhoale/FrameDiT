#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=Sky-MatLatteIMG-XL-256-1024
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/Sky-MatLatteIMG-XL-256-1024-Concat-%j.out
#SBATCH --error=slurm_log/Sky-MatLatteIMG-XL-256-1024-Concat-%j.err

source ~/.bashrc

conda activate latte

srun python train_with_img.py --config ./configs/sky/matlatteimg/sky_matlatteimg-256-1024-concat_train.yaml "$@"