#!/bin/bash
#SBATCH --job-name=running_eval1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

python run_scoring.py --run_name 42_imdb_run2 --batch_size 10