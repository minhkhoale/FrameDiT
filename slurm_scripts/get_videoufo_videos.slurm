#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=get-videoufo-videos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/dataset/%x-%j.out 
#SBATCH --error=slurm_log/dataset/%x-%j.err

source ~/.bashrc

conda activate tn

python scripts/videoufo/download.py