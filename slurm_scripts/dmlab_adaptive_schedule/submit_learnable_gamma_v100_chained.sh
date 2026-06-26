#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/train_learnable_gamma_v100.slurm"

CONFIGS=(
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_data_dependent_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_frequency_bin_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_scalar_gamma_train.yaml"
  "configs/dmlab_adaptive_schedule/learnable_gamma/dmlab_64_latte_learnable_timestep_gamma_train.yaml"
)

previous_job_id=""

for config_index in "${!CONFIGS[@]}"; do
  sbatch_args=(--parsable --export=ALL,CONFIG_INDEX="$config_index")
  if [[ -n "$previous_job_id" ]]; then
    sbatch_args+=(--dependency="after:${previous_job_id}")
  fi

  job_id="$(sbatch "${sbatch_args[@]}" "$WORKER")"
  echo "Submitted index ${config_index}: ${CONFIGS[$config_index]} as job ${job_id}"
  if [[ -n "$previous_job_id" ]]; then
    echo "  dependency: starts after job ${previous_job_id} starts"
  fi
  previous_job_id="$job_id"
done
