#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=sampleTaichi128-64_FusedMatLatte-M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_FusedMatLatte-M-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_FusedMatLatte-M-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-64/005-FusedMatLatte-M-64-256-2-concat-softmax-F64S3-taichi128-Compile-Amp-loadlatent-64frame --num-sampling-steps 250 --reverse --batch-size 8