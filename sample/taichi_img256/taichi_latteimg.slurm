#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large

#SBATCH --job-name=Taichi_latteimg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi_latteimg-%j.out
#SBATCH --error=slurm_log/sample/Taichi_latteimg-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/taichi_img256/001-LatteIMG-XL-2-F16S3-taichi_img256/ --num-sampling-steps 250