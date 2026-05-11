# Extract vectors pipeline

## 1. Benchmarks (`extract_vectors/download_dataset.py`)

Downloads **`stanfordnlp/imdb`** (via Hugging Face `datasets`) and writes balanced CSVs:

- **Train:** 2,000 positive + 2,000 negative (random per label; `--seed` configurable).
- **Val:** 20 positive + 20 negative from the **test** split (we use test as val).

Default output directory: **`benchmarks/`** (`train_pos.csv`, `train_neg.csv`, `val_pos.csv`, `val_neg.csv`). Sampling is random for now; a later improvement could pick more contrastive rows.

## 2. Steering vectors (`extract_vectors/extract_steer_vectors.py`)

Builds per-layer **positive** and **negative** direction vectors (mean hidden states over texts), either from the val CSVs or from a fixed pair when `--num-samples` is `0`.

**Examples:**

```bash
python extract_vectors/extract_steer_vectors.py --num-samples 20
python extract_vectors/extract_steer_vectors.py --num-samples 0   # uses "love" / "hate" only
```

- `--num-samples 0` → synthetic **love** (pos) / **hate** (neg); writes `steer_vectors/diffusion-val-n0.pt`.
- `--num-samples N` with `N > 0` → first `N` rows from `benchmarks/val_pos.csv` and `val_neg.csv`; writes `steer_vectors/diffusion-val-n{N}.pt`.

## 3. Val hyperparameter sweep (`extract_vectors/resteer_val_sweep_eval.py`)

Sweeps **α × layer** for a chosen vector file so you can pick **α** and **layer** before larger experiments. Outputs: **`extract_vectors/results_{tag}/`** (`scores.json`, `eval_scores.json`, heatmaps). The `{tag}` is inferred from the vector filename (e.g. `diffusion-val-n20.pt` → `results_20/`, `diffusion-val-n0.pt` → `results_0/`).

**Example:**

```bash
python extract_vectors/resteer_val_sweep_eval.py --direction negative --vectors steer_vectors/diffusion-val-n20.pt
```

**Evaluation:** sentiment and perplexity use `eval_dito.py` (same metrics as elsewhere in the repo).

**Steering:** the sweep uses **`resteer_v2_val`** in `llada/generate.py`—one identify → mask → one steered forward, **no** multi-step refill (lightweight probe, not the full iterative resteer setup).
