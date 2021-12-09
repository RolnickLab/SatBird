#!/bin/bash
#SBATCH --job-name=ebird
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=100Gb
#SBATCH --time=2-18:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

module load anaconda/3
conda activate ebird-env
python train2.py