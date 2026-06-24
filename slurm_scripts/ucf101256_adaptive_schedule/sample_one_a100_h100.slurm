#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101256-sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-a100|gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/ucf101256_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/ucf101256_adaptive_schedule/%x-%j.err

set -euo pipefail

EXP_DIR="${1:?Usage: sbatch sample_one_a100_h100.slurm <experiment_dir>}"
CKPT_STEP="${CKPT_STEP:-0500000}"
CONFIG="${EXP_DIR}/config.yaml"
CKPT="${EXP_DIR}/checkpoints/${CKPT_STEP}.pt"
EXP_NAME="$(basename "$EXP_DIR")"
SAVE_DIR="sample/ucf101256_adaptive_schedule/${EXP_NAME}/checkpoints_${CKPT_STEP}"

NUM_FVD_SAMPLES="${NUM_FVD_SAMPLES:-2048}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-6.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate latte

mkdir -p "$SAVE_DIR"

echo "Experiment dir: $EXP_DIR"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Save dir: $SAVE_DIR"
echo "Samples: $NUM_FVD_SAMPLES, method: $SAMPLE_METHOD, steps: $NUM_SAMPLING_STEPS, cfg: $CFG_SCALE, batch: $BATCH_SIZE"

python sample/sample_ddp.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --save_video_path "$SAVE_DIR" \
  --sample-method "$SAMPLE_METHOD" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --batch-size "$BATCH_SIZE" \
  --num-fvd-samples "$NUM_FVD_SAMPLES"
