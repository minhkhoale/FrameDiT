#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=dmlab-npz-to-mp4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm_log/dataset/%x-%j.out
#SBATCH --error=slurm_log/dataset/%x-%j.err

set -euo pipefail

SRC=/scratch/s224075134/temporal_diffusion/datasets/video/dmlab/training
DST=/scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/dmlab
PYTHON=/scratch/s224075134/.conda/envs/tn/bin/python

mkdir -p "$DST"

"$PYTHON" scripts/convert_npz_videos_to_mp4.py \
  --src "$SRC" \
  --dst "$DST" \
  --key video \
  --fps 30 \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
