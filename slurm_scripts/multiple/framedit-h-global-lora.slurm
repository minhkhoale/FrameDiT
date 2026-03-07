#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=FrameDiTH-Multiple-2-1-1-Global
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/multiple/%x-%j.out 
#SBATCH --error=slurm_log/multiple/%x-%j.err
source ~/.bashrc
export GPUS_PER_NODE=4


conda activate tn

accelerate launch --num_processes $GPUS_PER_NODE train_t2v_accelerate.py --config configs/multiple/framedith_t2v_global_train_lora.yaml