#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ucf_matlatteimg-xl-246-1024
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/ucf_matlatteimg-xl-246-1024-%j.out
#SBATCH --error=slurm_log/sample/ucf_matlatteimg-xl-246-1024-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/ucf101_img256/001-MatLatteIMG-XL-256-512-2-F16S3-ucf101_img256-Compile-Amp-loadpixel --num-sampling-steps 250 --reverse --batch-size 8 --resolution 256 --cfg-scale 7.0