#!/bin/bash
#SBATCH --job-name=ebird_baseline
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=34:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long

### this specifies the number of runs (we are doing 3 runs for now)
#SBATCH --array=1-3:1

# load conda environment
module load anaconda/3
conda activate satbird

# export keys for logging, etc,
export COMET_API_KEY=$COMET_API_KEY
export HYDRA_FULL_ERROR=1
# run training script
python train.py args.config=configs/SatBird-USA-summer/resnet18_RGBNIR_ENV_RM.yaml args.run_id=$SLURM_ARRAY_TASK_ID
