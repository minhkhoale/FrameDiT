#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --job-name=dmlab64-learnable-gamma
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
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_data_dependent_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_frequency_bin_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_scalar_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_timestep_gamma_train.yaml"
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
