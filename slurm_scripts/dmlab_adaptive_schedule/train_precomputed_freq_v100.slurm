#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --job-name=dmlab64-precomputed-freq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/dmlab_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/dmlab_adaptive_schedule/%x-%j.err

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate tn

CONFIGS=(
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.0_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.25_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.5_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.75_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_1.0_train_precomputed_freq.yaml"
)

CONFIG_INDEX="${CONFIG_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
CONFIG="${CONFIGS[$CONFIG_INDEX]}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

echo "Config index: $CONFIG_INDEX"
echo "Training config: $CONFIG"
srun python my_train_accelerate.py --config "$CONFIG"
