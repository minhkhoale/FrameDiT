#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101128-sample-fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_log/ucf101128_adaptive_schedule/%x-%A_%a.out
#SBATCH --error=slurm_log/ucf101128_adaptive_schedule/%x-%A_%a.out

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate latte

LABELS=(
  "gamma_0.0"
  "gamma_0.25"
  "gamma_0.5"
  "gamma_0.75"
  "gamma_1.0"
  "equalsnr"
)

LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
ROOT="results_adaptive_schedule/ucf101128"
OUT_ROOT="ucf101128-adaptive-schedule/fvd_samples"
NUM_FVD_SAMPLES="${NUM_FVD_SAMPLES:-2048}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-6.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"

read -r CONFIG CKPT SAVE_DIR <<< "$(python - "$LABEL" "$ROOT" "$OUT_ROOT" <<'PY'
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

label = sys.argv[1]
root = Path(sys.argv[2])
out_root = Path(sys.argv[3])


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def matches(cfg):
    if cfg.get("dataset") != "ucf101":
        return False
    if int(cfg.get("image_size", -1)) != 128:
        return False
    if cfg.get("model") != "MatLatte-B/64-256/2":
        return False

    if label == "equalsnr":
        return bool(cfg.get("equal_snr", False))

    if not bool(cfg.get("adaptive_frequency", False)):
        return False
    if not cfg.get("adaptive_frequency_power_path", None):
        return False
    target = float(label.split("_", 1)[1])
    gamma = as_float(cfg.get("adaptive_frequency_gamma"))
    return gamma is not None and abs(gamma - target) < 1e-9


candidates = []
for config_path in root.glob("*/config.yaml"):
    if "debug" in config_path.parts:
        continue
    cfg = OmegaConf.load(config_path)
    if matches(cfg):
        exp_dir = config_path.parent
        ckpt_dir = exp_dir / "checkpoints"
        checkpoints = sorted(
            ckpt_dir.glob("*.pt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
        )
        if checkpoints:
            candidates.append((checkpoints[-1].stat().st_mtime, exp_dir, checkpoints[-1]))

if not candidates:
    raise SystemExit(f"No completed experiment with checkpoints found for {label}")

_, exp_dir, ckpt = sorted(candidates, key=lambda item: item[0])[-1]
save_dir = out_root / label
print(exp_dir / "config.yaml", ckpt, save_dir)
PY
)"

mkdir -p "$SAVE_DIR"

echo "Sampling label: $LABEL"
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
