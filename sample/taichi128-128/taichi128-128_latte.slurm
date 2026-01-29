#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=sampleTaichi128-128_Latte-M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=slurm_log/sample/Taichi128-128_Latte-M-%j.out
#SBATCH --error=slurm_log/sample/Taichi128-128_Latte-M-%j.err

python -m tools.track_metrics --experiment-dir results/taichi128-128/000-Latte-M-2-F128S1-taichi128-Compile-Amp-loadlatent-128frame --num-sampling-steps 50 --sample-method ddim  --frequency 40000 --start-step 40000 --batch-size 8