#!/bin/bash
#SBATCH --job-name=download_summer_data
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

### this specifies the length of array
#SBATCH --array=1-7:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

keys=("bb14e853d0bb4a39aefed9e0efc92e5b" "66f928e7cdc843b789a96676c92b7cb2" "8ed8341639f44fc48c1a399564e9aa09" "07a3553db0f84e58b4776bf6a0541962" "a58d571b981a4ddc945405ea55e44dc6" "ab7327ce9c6a43cba55d4701209dcc0b" "e00fd9d71e7740f5a15510cd17566b98")

# load conda environment
module load miniconda/3
conda activate py38

echo ${keys[$SLURM_ARRAY_TASK_ID]} | planetarycomputer configure
python3 download_rasters_from_planetary_computer.py --index=$SLURM_ARRAY_TASK_ID --season="summer"