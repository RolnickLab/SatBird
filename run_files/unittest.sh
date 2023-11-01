#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=10Gb

module load anaconda/3
conda activate eco
pytest tests/data/test_data_files.py -k "test_nan_refl_image_values" -s