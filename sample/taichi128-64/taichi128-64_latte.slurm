#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=sampleTaichi128-64_Latte-M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=6
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128_Latte-M-%j.out
#SBATCH --error=slurm_log/sample/Taichi128_Latte-M-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-64/030-Latte-M-2-F64S3-taichi128-Compile-Amp-loadlatent-64frame --num-sampling-steps 250 --reverse --batch-size 8