#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=sample_Taichi128-32_FusedMatLatte-M-64-256-2-gated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128-32_FusedMatLatte-M-64-256-2-gated-%j.out
#SBATCH --error=slurm_log/sample/Taichi128-32_FusedMatLatte-M-64-256-2-gated-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-32/050-FusedMatLatte-M-64-256-2-gated-F32S3-taichi128-32f-no_load_latent --num-sampling-steps 250 --reverse