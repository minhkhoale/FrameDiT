#!/usr/bin/env bash
#SBATCH --job-name=Lattev2-BAIR16
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-v100
#SBATCH --cpus-per-task=5
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python train.py --config ./configs/bair/latte_train.yaml