#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=sample_Taichi128-32_FusedMatLatte-M-64-256-2-concat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=18
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128-32_FusedMatLatte-M-64-256-2-concat-%j.out
#SBATCH --error=slurm_log/sample/Taichi128-32_FusedMatLatte-M-64-256-2-concat-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-32/044-FusedMatLatte-M-64-256-2-concat-F32S3-taichi128-32f-novae --num-sampling-steps 250 --frequency 40000 --start-step 40000 --batch-size 16