#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --job-name=dmlab64-sample-videos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --array=0-5
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
)

EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
EXP_DIR="results_adaptive_schedule/dmlab64/${EXP_NAME}"
CONFIG="${EXP_DIR}/config.yaml"

CKPT_STEP="${CKPT_STEP:-0500000}"
CKPT="${EXP_DIR}/checkpoints/${CKPT_STEP}.pt"
SAVE_DIR="${SAVE_DIR:-sample/dmlab_adaptive_schedule/${EXP_NAME}/checkpoints_${CKPT_STEP}}"

NUM_FVD_SAMPLES="${NUM_FVD_SAMPLES:-2048}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-6.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"
SEED="${SEED:-1024}"
USE_FP16="${USE_FP16:-true}"

# DMLab uses extras=3 action conditioning. By default this samples real action
# sequences from ACTION_PATH. Set ACTION_LABEL to fix one action, or
# ACTION_SEQUENCE to pass 16 labels.
ACTION_PATH="${ACTION_PATH:-/scratch/s224075134/temporal_diffusion/datasets/video/dmlab/training}"
ACTION_LABEL="${ACTION_LABEL:--1}"
ACTION_SEQUENCE="${ACTION_SEQUENCE:-}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Save dir: $SAVE_DIR"
echo "Samples: $NUM_FVD_SAMPLES, method: $SAMPLE_METHOD, steps: $NUM_SAMPLING_STEPS, cfg: $CFG_SCALE, batch: $BATCH_SIZE"
echo "Action path: $ACTION_PATH"
echo "Action label: $ACTION_LABEL"
echo "Action sequence: $ACTION_SEQUENCE"

srun python sample/sample_ddp.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --save_video_path "$SAVE_DIR" \
  --sample-method "$SAMPLE_METHOD" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --batch-size "$BATCH_SIZE" \
  --num-fvd-samples "$NUM_FVD_SAMPLES" \
  --seed "$SEED" \
  --use-fp16 "$USE_FP16" \
  --action-path "$ACTION_PATH" \
  --action-label "$ACTION_LABEL" \
  --action-sequence "$ACTION_SEQUENCE"
