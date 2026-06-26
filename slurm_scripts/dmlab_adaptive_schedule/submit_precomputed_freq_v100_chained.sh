#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/train_precomputed_freq_v100.slurm"

CONFIGS=(
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.0_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.25_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.5_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_0.75_train_precomputed_freq.yaml"
  "configs/dmlab_adaptive_schedule/precomputed_freq/dmlab_64_latte_gamma_1.0_train_precomputed_freq.yaml"
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
