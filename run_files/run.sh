#!/bin/bash
#SBATCH --job-name=ebird
#SBATCH --output=job_output.txts
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --mem=100Gb
#SBATCH --time=2-18:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err





#!/bin/bash

##########################
# This file is a template. Fill in the blanks with python.
#
# To stop this job, get the jobid using
#   squeue -u <username>
# then cancel the job with
#   scancel <jobid>
##########################
#source you virtualenv

module load anaconda/3

GPUS=1
echo "Number of GPUs: "${GPUS}
WRAP="python train2.py args.config=configs/base_RGB_RM-satmae-ENV.yaml"
#WRAP='python test2.py'
JOBNAME="correction_ecosys"
LOG_FOLDER="/home/mila/a/amna.elmustafa/ecosys_logs"
echo ${WRAP}
echo "Log Folder:"${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}
# print out Slurm Environment Variables
echo "
Slurm Environment Variables:
- SLURM_JOBID=$SLURM_JOBID
- SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
- SLURM_NNODES=$SLURM_NNODES
- SLURMTMPDIR=$SLURMTMPDIR
- SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR
"

# slurm doesn't source .bashrc automatically
source ~/.bashrc

project_dir="/network/scratch/a/amna.elmustafa/hager/ecosystem-embedding"
echo "Setting directory to: $project_dir"
cd $project_dir

# list out some useful information
echo "
Basic system information:
- Date: $(date)
- Hostname: $(hostname)
- User: $USER
- pwd: $(pwd)
"
conda activate ebird.2

#{content}

export CUDA_VISIBLE_DEVICES=0,1
export COMET_API_KEY="9PY4gOZFYKFRPw5xSCtDpdM7H"
export COMET_WORKSPACE="amnaalmgly"

sbatch --output=${LOG_FOLDER}/%j.out --error=${LOG_FOLDER}/%j.err \
    --nodes=1 --ntasks-per-node=1 --time=2-00:00:00 --mem=40G \
    --partition=long --cpus-per-task=4 \
    --gres=gpu:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"


echo "All jobs launched!"
echo "Waiting for child processes to finish..."
wait
echo "Done!"






