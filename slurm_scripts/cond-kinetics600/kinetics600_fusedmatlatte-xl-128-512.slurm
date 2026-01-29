#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=condkinetics600_fusedmatlatte-xl-128-512_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --mem=256G
#SBATCH --constraint=gpu-h100
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/condkinetics600_fusedmatlatte-xl-128-512_train-%j.out
#SBATCH --error=slurm_log/condkinetics600_fusedmatlatte-xl-128-512_train-%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/cond-kinetics600/fusedmatlatte/kinetics600_fusedmatlatte-xl-pixel_train.yaml "$@"