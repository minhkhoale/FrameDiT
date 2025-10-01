#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=Taichi128_MatLatte-32-256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --constraint=gpu-h200
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_MatLatte-32-256-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_MatLatte-32-256-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/006-MatLatte-M-32-256-2-F16S3-taichi128 --num-sampling-steps 250 --wandb-run-id vdm/b7yn3mcw