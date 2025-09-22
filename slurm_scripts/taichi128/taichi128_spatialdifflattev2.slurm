#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_spatialdifflattev2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=5
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/taichi128_spatialdifflattev2-%j.out 
#SBATCH --error=slurm_log/taichi128_spatialdifflattev2-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference_v2.py --config ./configs/taichi128/spatialdifflattev2/taichi128_spatialdifflattev2_train.yaml "$@"