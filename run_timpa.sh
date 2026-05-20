python3 run_timpa.py \
  --run-name "sample_timpa" \
  --random-state 42 \
  --steer-vector-path "steer_vectors/diffusion-val-n20.pt" \
  --batch-size 4 \
  --resteer-steps 32 \
  --refill-steps 5 6 \
  --sampling-temp 1.0 \
  --identify-temp 0.5
