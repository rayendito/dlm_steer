# Cats vs Dogs Steering Pipeline

This folder adds an end-to-end pipeline for:

1. Generating a synthetic cats/dogs dataset using Appendix A style prompts from ExpertLens.
2. Post-filtering and splitting data into train/val/test.
3. Extracting steering vectors for LLaDA.
4. Running greedy ablations on sentence length bin (N), detection temperature (T), steering steps (k), refill steps (u).
5. Reporting target-animal probability, perplexity, and harmonic composite score.

## Scripts

- `generate_cats_dogs_synth.py`
  - Generates oversampled raw data with Mistral-7b-Instruct-v0.2.
- `postprocess_cats_dogs.py`
  - Filters duplicates/weird rows and creates exact 1500+1500 plus 80/10/10 splits.
- `extract_cats_dogs_vectors.py`
  - Builds per-layer positive/negative mean vectors from train split.
- `run_cats_dogs_greedy_ablation.py`
  - Runs greedy search and writes full tables + qualitative examples.
- `qsub_templates/`
  - PBS templates for generation, postprocess, vector extraction, and ablation.

## Typical flow

1) Generate raw:

```bash
python cats_dogs/generate_cats_dogs_synth.py \
  --output-dir benchmarks/cats_dogs \
  --per-class-target 1500 \
  --oversample-factor 1.6 \
  --batch-prompts 120
```

2) Postprocess/split:

```bash
python cats_dogs/postprocess_cats_dogs.py \
  --input-jsonl benchmarks/cats_dogs/raw_generations.jsonl \
  --output-dir benchmarks/cats_dogs \
  --per-class-target 1500
```

3) Extract vectors:

```bash
python cats_dogs/extract_cats_dogs_vectors.py \
  --train-csv benchmarks/cats_dogs/train.csv \
  --output-path steer_vectors/cats_dogs.pt
```

4) Greedy ablation:

```bash
python cats_dogs/run_cats_dogs_greedy_ablation.py \
  --dataset-csv benchmarks/cats_dogs/test.csv \
  --vectors-path steer_vectors/cats_dogs.pt \
  --output-dir cats_dogs/results
```
