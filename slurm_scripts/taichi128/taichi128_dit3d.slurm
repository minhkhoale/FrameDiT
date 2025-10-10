#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=Taichi128-DiT3D
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/dit3d/taichi128_dit3d_train.yaml "$@"