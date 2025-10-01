#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=FusedMatLatte-M-64-256-2-sum-F16S3-taichi128
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/FusedMatLatte-M-64-256-2-sum-F16S3-taichi128-%j.out
#SBATCH --error=slurm_log/sample/FusedMatLatte-M-64-256-2-sum-F16S3-taichi128-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/030-FusedMatLatte-M-64-256-2-sum-F16S3-taichi128 --num-sampling-steps 250 --reverse