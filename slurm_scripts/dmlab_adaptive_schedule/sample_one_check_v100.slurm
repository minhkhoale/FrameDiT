#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --job-name=dmlab64-sample-one
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-v100
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=slurm_log/dmlab_adaptive_schedule/%x-%j.out
#SBATCH --error=slurm_log/dmlab_adaptive_schedule/%x-%j.err

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate tn

EXP_NAME="${EXP_NAME:-006-LatteV2-B-1-F16S1-dmlab64-Compile-Amp-loadlatent}"
EXP_DIR="results_adaptive_schedule/dmlab64/${EXP_NAME}"
CONFIG="${EXP_DIR}/config.yaml"

CKPT_STEP="${CKPT_STEP:-0500000}"
CKPT="${EXP_DIR}/checkpoints/${CKPT_STEP}.pt"
SAVE_DIR="${SAVE_DIR:-sample/dmlab_adaptive_schedule/check_one/${EXP_NAME}/checkpoints_${CKPT_STEP}}"
ACTION_PATH="${ACTION_PATH:-/scratch/s224075134/temporal_diffusion/datasets/video/dmlab/training}"

NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-6.0}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"
SEED="${SEED:-1024}"
USE_FP16="${USE_FP16:-true}"

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
echo "Action path: $ACTION_PATH"

srun python sample/sample_ddp.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --save_video_path "$SAVE_DIR" \
  --sample-method "$SAMPLE_METHOD" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --batch-size 1 \
  --num-fvd-samples 1 \
  --seed "$SEED" \
  --use-fp16 "$USE_FP16" \
  --action-path "$ACTION_PATH"
