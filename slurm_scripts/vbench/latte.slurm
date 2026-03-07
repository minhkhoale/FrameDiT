#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=vbench-sample-latte
#SBATCH --partition=gpu-large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus=3
#SBATCH --constraint=gpu-h100
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/vbench/%x-%j.out 
#SBATCH --error=slurm_log/vbench/%x-%j.err

conda activate tn

srun python sample/sample_t2x.py --config configs/t2v/sample/lattet2v_vbench.yaml