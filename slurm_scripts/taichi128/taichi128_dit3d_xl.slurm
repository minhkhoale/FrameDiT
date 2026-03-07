#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_dit3d-XL
#SBATCH --nodes=1
#SBATCH --partition=gpu-large
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128/%x-%j.out 
#SBATCH --error=slurm_log/taichi128/%x-%j.err

source ~/.bashrc

conda activate tn

srun python my_train.py --config ./configs/taichi128/dit3d/taichi128_dit3d_xl_train.yaml "$@"