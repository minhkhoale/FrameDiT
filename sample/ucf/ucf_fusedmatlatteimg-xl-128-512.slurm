#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=ucf_fusedmatlatteimg-xl-128-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=7
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/ucf_fusedmatlatteimg-xl-128-512-%j.out
#SBATCH --error=slurm_log/sample/ucf_fusedmatlatteimg-xl-128-512-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/ucf101_img256/000-FusedMatLatteIMG-XL-128-512-2-concat-F16S3-ucf101_img256-ImageNetPretrained-Compile-Amp-loadpixel --num-sampling-steps 250 --reverse --batch-size 8 --resolution 256 --cfg-scale 7.0