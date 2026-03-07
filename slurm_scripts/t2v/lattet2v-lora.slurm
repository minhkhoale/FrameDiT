#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=LatteT2V-LORA
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/t2v/%x-%j.out 
#SBATCH --error=slurm_log/t2v/%x-%j.err
source ~/.bashrc

conda activate tn

srun python train_t2v_lora.py --config configs/t2v/lattet2v_lora.yaml