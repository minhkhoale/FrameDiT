#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=sampleTaichi128-32_MatLatte-M-64-256
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_MatLatte-M-64-256-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_MatLatte-M-64-256-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-32/050-MatLatte-M-64-256-2-F32S3-taichi128-32f-no_load_latent --num-sampling-steps 250 --reverse