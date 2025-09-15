#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=taichi128_difflattev1-noise-correlation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --constraint=gpu-v100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=5
#SBATCH --time=60:00:00
#SBATCH --output=slurm_log/taichi128_difflattev1-noise-correlation-%j.out
#SBATCH --error=slurm_log/taichi128_difflattev1-noise-correlation-%j.err

source ~/.bashrc

conda activate latte

srun python my_train_difference_v1.py --config ./configs/taichi128/difflatte_v1/taichi128_difflattev1-noise-correlation_train.yaml "$@"