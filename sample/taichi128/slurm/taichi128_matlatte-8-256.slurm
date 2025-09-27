#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=Taichi128_MatLatte-8-256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_MatLatte-8-256-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_MatLatte-8-256-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/005-MatLatte-M-8-256-2-F16S3-taichi128 --num-sampling-steps 250 --wandb-run-id vdm/drikknzv