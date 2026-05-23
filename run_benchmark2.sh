#!/bin/bash
#SBATCH --job-name=runbench2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

python run_benchmark.py \
  --dataset_path benchmarks/imdb \
  --batch_size 64 \
  --random_seed 42