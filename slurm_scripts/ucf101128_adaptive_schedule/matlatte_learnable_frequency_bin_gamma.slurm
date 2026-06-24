#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101128-matlatte-learnable-frequency-bin-gamma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/ucf101128_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/ucf101128_adaptive_schedule/%x-%j.out

source /scratch/s224075134/.conda/etc/profile.d/conda.sh

conda activate latte

srun python my_train_accelerate.py --config configs/ucf101_adaptive_schedule/learnable_gamma/ucf101_128_matlatte_learnable_frequency_bin_gamma_train.yaml
