#!/usr/bin/env bash
#SBATCH --qos=batch-short
#SBATCH --job-name=get-panda70m-videos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --output=slurm_log/dataset/%x-%j.out 
#SBATCH --error=slurm_log/dataset/%x-%j.err

source ~/.bashrc

cd /scratch/s224075134/temporal_diffusion/datasets/video/panda70m/Panda-70M/dataset_dataloading/video2dataset
conda activate pd1

python test.py --csv /scratch/s224075134/temporal_diffusion/datasets/video/panda70m/panda70m_training_2m.csv