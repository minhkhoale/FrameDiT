#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=ffs_matlatteimg-xl-256-1024-softmax
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=3
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/ffs_matlatteimg-xl-256-1024-softmax-%j.out
#SBATCH --error=slurm_log/sample/ffs_matlatteimg-xl-256-1024-softmax-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/ffs_img256/003-MatLatteIMG-XL-256-1024-2-softmax-F16S3-ffs_img256-Compile-Amp-loadpixel --num-sampling-steps 250 --reverse --batch-size 16