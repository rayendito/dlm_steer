#!/bin/bash
#SBATCH --job-name=running
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

RANDOM_STATE=42
BASE_RUN_NAME="imdb_run"
RUN_NAME="${RANDOM_STATE}_${BASE_RUN_NAME}"

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "benchmarks/imdb" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "extract_vectors/steer_vectors/diffusion-val-n20.pt" \
  --steer-alpha 500 \
  --steer-layers 16 25 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 5 10 15 \
  --sampling-temp 0.5 \
  --identify-temp 0.5

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "benchmarks/imdb" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "extract_vectors/steer_vectors/diffusion-val-n20.pt" \
  --steer-alpha 500 \
  --steer-layers 16 25 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 10 \
  --sampling-temp 0.25 1 \
  --identify-temp 0.5

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "benchmarks/imdb" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "extract_vectors/steer_vectors/diffusion-val-n20.pt" \
  --steer-alpha 500 \
  --steer-layers 16 25 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 10 \
  --sampling-temp 0.5 \
  --identify-temp 0.25 1
