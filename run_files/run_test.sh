#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --output=job_output_test_baselines.txt
#SBATCH --error=job_error_test_baselines.txt
#SBATCH --ntasks=1
#SBATCH --time=5:59:00
#SBATCH --mem-per-cpu=20Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load anaconda/3
conda activate eco

python test.py args.config=configs/SatBird-USA-summer/resnet18_RGB.yaml
python test.py args.config=configs/SatBird-USA-summer/resnet18_RGB_ENV.yaml
python test.py args.config=configs/SatBird-USA-summer/resnet18_RGBNIR.yaml
python test.py args.config=configs/SatBird-USA-summer/resnet18_RGBNIR_ENV.yaml
python test.py args.config=configs/SatBird-USA-summer/resnet18_RGBNIR_ENV_RM.yaml
