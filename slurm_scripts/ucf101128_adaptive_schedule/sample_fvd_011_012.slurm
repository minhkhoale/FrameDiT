#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf101128-011-012-fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_log/ucf101128_adaptive_schedule/%x-%A_%a.out
#SBATCH --error=slurm_log/ucf101128_adaptive_schedule/%x-%A_%a.err

set -euo pipefail

source /scratch/s224075134/.conda/etc/profile.d/conda.sh
conda activate latte

EXPERIMENTS=(
  "011-MatLatte-B-64-256-2-F16S3-ucf101128-Compile-Amp-loadlatent"
  "012-MatLatte-B-64-256-2-F16S3-ucf101128-gaussian_diffusion-equalsnr_channelwise_calibrated-Compile-Amp-loadlatent"
)

EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
EXP_DIR="results_adaptive_schedule/ucf101128/${EXP_NAME}"
CONFIG="${EXP_DIR}/config.yaml"
CKPT="${EXP_DIR}/checkpoints/0500000.pt"
SAVE_DIR="sample/ucf101128_adaptive_schedule/${EXP_NAME}/checkpoints_0500000"
REAL_DATA_PATH="${REAL_DATA_PATH:-$HOME/scratch/temporal_diffusion/datasets/video/ucf101/reconstruction_128}"

NUM_FVD_SAMPLES="${NUM_FVD_SAMPLES:-2048}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-6.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddim}"
SEED="${SEED:-1024}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

if [[ ! -d "$REAL_DATA_PATH" ]]; then
  echo "Missing real data path: $REAL_DATA_PATH" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Save dir: $SAVE_DIR"
echo "Real data path: $REAL_DATA_PATH"
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

for REAL_SAMPLE_FACTOR in 1 2 3 4 5 6; do
  RESULT_FILE="${SAVE_DIR}/fvd_real_sample_factor_${REAL_SAMPLE_FACTOR}.json"
  echo "Calculating FVD with real-sample-factor=${REAL_SAMPLE_FACTOR}; result_file=${RESULT_FILE}"
  python tools/my_cal_metrics_for_dataset.py \
    --real_data_path "$REAL_DATA_PATH" \
    --fake_data_path "$SAVE_DIR" \
    --resolution 128 \
    --real-sample-factor "$REAL_SAMPLE_FACTOR" \
    --verbose \
    --seed "$SEED" \
    --result_file "$RESULT_FILE"
done
