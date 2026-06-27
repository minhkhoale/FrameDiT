#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --job-name=dmlab64-am-fvd-v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-10
#SBATCH --output=slurm_log/dmlab_adaptive_schedule/%x-%A_%a.out
#SBATCH --error=slurm_log/dmlab_adaptive_schedule/%x-%A_%a.err

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate tn

EXPERIMENTS=(
  "001-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "002-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "003-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "004-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "005-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "006-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "007-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "008-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "009-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "010-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
  "011-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent"
)

EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
FAKE_DATA_PATH="sample/dmlab_adaptive_schedule/${EXP_NAME}/checkpoints_0500000_action_matched-seed-1024"
RESULT_FILE="${FAKE_DATA_PATH}/action_matched_fvd.json"

if [[ ! -d "$FAKE_DATA_PATH" ]]; then
  echo "Missing fake data path: $FAKE_DATA_PATH" >&2
  exit 1
fi

echo "Experiment: $EXP_NAME"
echo "Fake data path: $FAKE_DATA_PATH"
echo "Result file: $RESULT_FILE"

python tools/cal_dmlab_action_matched_fvd.py \
  --fake_data_path "$FAKE_DATA_PATH" \
  --max_size 2048 \
  --batch_size 64 \
  --seed 21 \
  --result_file "$RESULT_FILE"
