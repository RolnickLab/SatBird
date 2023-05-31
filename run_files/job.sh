#!/bin/bash
#SBATCH --job-name=ebird_1st
#SBATCH --output=job_output_test.txt
#SBATCH --error=job_error_test.txt
#SBATCH --ntasks=1
#SBATCH --time=10:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module load miniconda/3
conda activate eco
export COMET_API_KEY=$COMET_API_KEY
export HYDRA_FULL_ERROR=1
python train.py  ++auto_lr_find="False" args.config=configs/base.yaml args.run_id=1
