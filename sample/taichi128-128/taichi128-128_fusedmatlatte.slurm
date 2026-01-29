#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=sampleTaichi128-128_FusedMatLatte-M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=18
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi128-128_FusedMatLatte-M-%j.out
#SBATCH --error=slurm_log/sample/Taichi128-128_FusedMatLatte-M-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-128/001-FusedMatLatte-M-64-256-2-concat-softmax-F128S3-taichi128-Compile-Amp-loadlatent-128frame --num-sampling-steps 250 --frequency 40000 --start-step 40000 --batch-size 4