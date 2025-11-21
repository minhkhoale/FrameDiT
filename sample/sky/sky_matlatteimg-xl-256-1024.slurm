#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=sky_matlatteimg-xl-256-1024
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/sky_matlatteimg-xl-256-1024-%j.out
#SBATCH --error=slurm_log/sample/sky_matlatteimg-xl-256-1024-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/sky_img256/003-MatLatteIMG-XL-256-1024-2-softmax-F16S3-sky_img256-Compile-Amp-loadpixel --num-sampling-steps 250 --reverse --batch-size 16 --resolution 256