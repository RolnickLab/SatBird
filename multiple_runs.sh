#!/bin/bash
#SBATCH --job-name=ebird_baseline
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=1:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long

### this specifies the number of runs (we are doing 3 runs for now)
#SBATCH --array=1-3:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

# load conda environment
module load miniconda/3
conda activate eco

# export keys for logging, etc,
export COMET_API_KEY=$USER_COMET_COMET_API_KEY
export HYDRA_FULL_ERROR=1

# run training script
python train2.py args.config=configs/base.yaml args.run_id=$SLURM_ARRAY_TASK_ID