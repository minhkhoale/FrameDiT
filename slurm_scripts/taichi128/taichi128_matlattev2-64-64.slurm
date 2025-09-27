#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=MatLattev2-taichi128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-l40s
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/%j.out 
#SBATCH --error=slurm_log/%j.err

source ~/.bashrc

conda activate latte

srun python my_train.py --config ./configs/taichi128/matlattev2/taichi128_matlattev2-64-64_train.yaml "$@"