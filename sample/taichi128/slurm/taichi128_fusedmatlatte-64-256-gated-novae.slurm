#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=FusedMatLatte-M-64-256-2-gated-F16S3-taichi128-novae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/FusedMatLatte-M-64-256-2-gated-F16S3-taichi128-novae-%j.out
#SBATCH --error=slurm_log/sample/FusedMatLatte-M-64-256-2-gated-F16S3-taichi128-novae-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/037-FusedMatLatte-M-64-256-2-gated-F16S3-taichi128-novae --num-sampling-steps 250 --wandb-run-id vdm/qkz14yeq