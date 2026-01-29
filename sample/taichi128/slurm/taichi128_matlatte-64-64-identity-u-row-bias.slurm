#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=Taichi128_MatLatte-M-64-64-2-identity-u-row-bias
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_MatLatte-M-64-64-2-identity-u-row-bias-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_MatLatte-M-64-64-2-identity-u-row-bias-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/027-MatLatte-M-64-64-2-identity-u-row-bias-F16S3-taichi128/ --num-sampling-steps 250 --wandb-run-id vdm/qau0eo6y --reverse