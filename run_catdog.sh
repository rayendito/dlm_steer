#!/bin/bash
#SBATCH --job-name=run1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

RANDOM_STATE=42
BASE_RUN_NAME="cats_dogs_run2"
RUN_NAME="${RANDOM_STATE}_${BASE_RUN_NAME}"
BENCHMARK="benchmarks/cats_dogs"
VECTOR_PATH="steer_vectors/diffusion-catdog-n10.pt"

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "$BENCHMARK" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "$VECTOR_PATH" \
  --steer-alpha 75 \
  --steer-layers 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 5 10 15 \
  --sampling-temp 0.5 \
  --identify-temp 0.5

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "$BENCHMARK" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "$VECTOR_PATH" \
  --steer-alpha 75 \
  --steer-layers 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 10 \
  --sampling-temp 0.25 1 \
  --identify-temp 0.5

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "$BENCHMARK" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "$VECTOR_PATH" \
  --steer-alpha 75 \
  --steer-layers 31 \
  --batch-size 8 \
  --resteer-steps 5 \
  --refill-steps 10 \
  --sampling-temp 0.5 \
  --identify-temp 0.25 1
