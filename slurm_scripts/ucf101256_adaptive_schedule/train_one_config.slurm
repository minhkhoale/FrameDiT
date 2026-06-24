#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101256-train-one
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/ucf101256_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/ucf101256_adaptive_schedule/%x-%j.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch train_one_config.slurm <config.yaml>}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate latte

echo "Training config: $CONFIG"
srun python my_train_accelerate.py --config "$CONFIG"
