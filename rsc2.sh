#!/bin/bash
#SBATCH --job-name=running_eval2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

python run_scoring.py --run_name 42_imdb_run_long --batch_size 10