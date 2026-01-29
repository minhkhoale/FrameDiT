#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=sample_Taichi128_MatLatte-16-256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_MatLatte-16-256-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_MatLatte-16-256-%j.err

srun python -m tools.track_metrics --experiment-dir results/taichi128/050-MatLatte-M-16-256-2-F16S3-taichi128-no_load_latent --num-sampling-steps 250 --reverse