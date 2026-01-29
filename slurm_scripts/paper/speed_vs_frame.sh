#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=speed_vs_frame
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/paper/speed_vs_frame-%j.out
#SBATCH --error=slurm_log/paper/speed_vs_frame-%j.err

source ~/.bashrc

conda activate latte

cd paper

python speed_vs_frame.py