#!/bin/bash
#SBATCH --job-name=download_range_maps
#SBATCH --output=job_output_RM.txt
#SBATCH --error=job_error_RM.txt
#SBATCH --ntasks=1
#SBATCH --time=04:59:00
#SBATCH --mem-per-cpu=10Gb
#SBATCH --cpus-per-task=8
#SBATCH --partition=long-cpu

### this specifies the length of array
#SBATCH --array=9-16:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

# load conda environment
module load miniconda/3
conda activate py38

python3 get_range_maps.py --index=$SLURM_ARRAY_TASK_ID
