#!/bin/bash
#SBATCH --job-name=download_winter_data
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

### this specifies the length of array
#SBATCH --array=1-4:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

keys=("9d32d41bb1cb4b18838f68948cbd3b77" "f355acf423fd456790fa67cf2a173aa5" "2d1bbb5340b542d89b4c6b634f26119d" "385e851a7c284677bd9d911b2e178637")


# load conda environment
module load miniconda/3
conda activate py38

echo ${keys[$SLURM_ARRAY_TASK_ID]} | planetarycomputer configure
python3 download_rasters_from_planetary_computer.py --index=$SLURM_ARRAY_TASK_ID --season="winter"