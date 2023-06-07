#!/bin/bash
#SBATCH --job-name=ebird_baseline
#SBATCH --output=job_output2.txt
#SBATCH --error=job_error2.txt
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long

### this specifies the number of runs (we are doing 3 runs for now)
#SBATCH --array=1-3:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

# load conda environment
module load anaconda/3
conda activate ebird-env

# export keys for logging, etc,
export COMET_API_KEY="WUET2WpClgOsQfj79XFjAU4ce"
export HYDRA_FULL_ERROR=1

# run training script
python train.py args.config=configs/kenya_transfer_learning.yaml args.run_id=$SLURM_ARRAY_TASK_ID