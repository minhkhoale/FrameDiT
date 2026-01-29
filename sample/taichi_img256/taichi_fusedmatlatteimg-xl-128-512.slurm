#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --partition=gpu-large
#SBATCH --job-name=Taichi_fusedmatlatteimg-xl-128-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu-h100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/Taichi_fusedmatlatteimg-xl-128-512-%j.out
#SBATCH --error=slurm_log/sample/Taichi_fusedmatlatteimg-xl-128-512-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/taichi_img256/009-FusedMatLatteIMG-XL-128-512-2-concat-F16S3-taichi_img256-Compile-Amp-loadpixel --num-sampling-steps 50 --reverse --batch-size 8 --resolution 256 --sample-method ddim