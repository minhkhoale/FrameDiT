#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101256-precomputed-freq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm_log/ucf101256_adaptive_schedule/%x-%A_%a.out
#SBATCH --error=slurm_log/ucf101256_adaptive_schedule/%x-%A_%a.err

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate latte

CONFIGS=(
  "configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_0.0_train_precomputed_freq.yaml"
  "configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_0.25_train_precomputed_freq.yaml"
  "configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_0.5_train_precomputed_freq.yaml"
  "configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_0.75_train_precomputed_freq.yaml"
  "configs/ucf101_256_adaptive_schedule/precomputed_freq/ucf101_256_matlatte_gamma_1.0_train_precomputed_freq.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

echo "Training config: $CONFIG"
srun python my_train_accelerate.py --config "$CONFIG"
