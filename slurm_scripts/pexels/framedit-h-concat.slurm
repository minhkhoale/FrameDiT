#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=FrameDiTH-Pexels-2-1-1-Concat
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/pexels/%x-%j.out 
#SBATCH --error=slurm_log/pexels/%x-%j.err
source ~/.bashrc
export GPUS_PER_NODE=4


conda activate tn

accelerate launch --num_processes $GPUS_PER_NODE train_t2v_accelerate.py --config configs/pexels/framedith_t2v_train_matattn_concat.yaml