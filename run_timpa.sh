RANDOM_STATE=42
BASE_RUN_NAME="imdb_run2"
RUN_NAME="${RANDOM_STATE}_${BASE_RUN_NAME}"

python run_timpa.py \
  --run-name "$RUN_NAME" \
  --dataset-path "benchmarks/imdb" \
  --random-state "$RANDOM_STATE" \
  --steer-vector-path "extract_vectors/steer_vectors/diffusion-val-n20.pt" \
  --steer-alpha 500 \
  --steer-layers 16 25 31 \
  --batch-size 2 \
  --resteer-steps 3 \
  --refill-steps 10 \
  --sampling-temp 1.0 \
  --identify-temp 0.5
