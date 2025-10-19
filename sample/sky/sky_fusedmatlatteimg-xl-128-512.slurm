#!/bin/bash
#!/bin/bash
#SBATCH --qos=batch-short
#SBATCH --job-name=sky_fusedmatlatteimg-xl-128-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=a100-m-02
#SBATCH --gpus=8
#SBATCH --constraint=gpu-a100
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/sample/sky_fusedmatlatteimg-xl-128-512-%j.out
#SBATCH --error=slurm_log/sample/sky_fusedmatlatteimg-xl-128-512-%j.err

srun python -m tools.track_metrics --experiment-dir results_img/sky_img256/000-FusedMatLatteIMG-XL-128-512-2-concat-F16S3-sky_img256-Compile-Amp --num-sampling-steps 250 --reverse --batch-size 8 --resolution 256