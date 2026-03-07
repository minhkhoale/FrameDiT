#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=FrameDiTH-T2V-32-Difference-Gate-LORA
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/t2v/%x-%j.out 
#SBATCH --error=slurm_log/t2v/%x-%j.err
source ~/.bashrc

conda activate tn

srun python train_t2v.py --config configs/t2v/framedit_h_difference_gate_lora.yaml