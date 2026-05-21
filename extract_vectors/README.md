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

### New options

- `--concept-pair sentiment|cat-dog`
- `--sample-seed <int>` random row sampling seed
- `--source-split val|train` (for requested split flip, use `val` for vector construction)
- `--pair-order cat,dog|dog,cat` when `--concept-pair cat-dog`

If `--num-samples 0` and `--concept-pair cat-dog`, extraction uses token-only anchors in the selected order (e.g. `cat,dog` means cat minus dog direction).

### Sentence-count ablation (3 random seeds)

Run:

```bash
python extract_vectors/run_vector_count_ablation.py \
  --concept-pair cat-dog \
  --pair-order cat,dog \
  --vector-source-split val
```

This runs `n in {0,5,10,15,20,30,40,50}` over seeds `{41,42,43}`.

## 3. Val hyperparameter sweep (`extract_vectors/resteer_val_sweep_eval.py`)

Sweeps **α × layer** for a chosen vector file so you can pick **α** and **layer** before larger experiments. Outputs: **`extract_vectors/results_{pos|neg}_{tag}/`** (`scores.json`, `eval_scores.json`, heatmaps). `{pos|neg}` comes from `--direction` (`positive` → `pos`, `negative` → `neg`). `{tag}` is inferred from the vector filename (e.g. `diffusion-val-n20.pt` → `20` → `results_neg_20/` with `--direction negative`).

**Example:**

```bash
python extract_vectors/resteer_val_sweep_eval.py --direction negative --vectors steer_vectors/diffusion-val-n20.pt
```

**Evaluation:** sentiment and perplexity use `eval_dito.py` (same metrics as elsewhere in the repo).

**Steering:** the sweep calls **`resteer_v2`** in `llada/generate.py` (identify → mask → `refill_steps` steered refills per outer `resteer_steps` iteration). **`RESTEER_STEPS`=1**, **`REFILL_STEPS`=1**, **`PERPLEXITY_THRESHOLD`=10000**, **`SENTIMENT_THRESHOLD`=0.1**, and related knobs live at the top of `resteer_val_sweep_eval.py`.

## 4. Merge pos + neg summaries (`extract_vectors/merge_eval_results.py`)

After you have **paired** folders `results_pos_{tag}/` and `results_neg_{tag}/` (same `tag`, e.g. `0`, `1`, `20`), run the merge script. It auto-detects which tags have both directions, reads each side’s `eval_scores.json`, copies per-direction top‑5 harmonic-mean rows (unique layer), pairs those ranks, and builds **cross-direction** top‑5 over the **full** shared `(layer, α)` grid (not limited to those per-direction top‑5s), again **one α per layer** chosen by greedy cross harmonic mean.

**Example:**

```bash
python extract_vectors/merge_eval_results.py
```

---

## Top 5 by vector variation (`n`)

Snapshot from **`extract_vectors/results.json`** (per-direction top‑5: unique layers; harmonic mean = sentiment vs. 1 − robust-normalized perplexity, as in the sweep). **cross_hm** = harmonic mean of pos- and neg-direction harmonic scores on the **same** `(layer, α)`; the cross table is **top 5 over the full grid intersection with unique layers** (greedy by `cross_hm`).

### n=0

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 0 | 10.0 | 0.3403 | 0.3830 | 0.3061 |
| 2 | 26 | 50.0 | 0.1433 | 0.3055 | 0.0936 |
| 3 | 10 | 5.0 | 0.1422 | 0.3305 | 0.0906 |
| 4 | 23 | 0.01 | 0.1358 | 0.2727 | 0.0904 |
| 5 | 24 | 0.1 | 0.1329 | 0.2290 | 0.0936 |

### n=1

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 14 | 100.0 | 0.2398 | 0.3748 | 0.1763 |
| 2 | 0 | 50.0 | 0.2172 | 0.1571 | 0.3520 |
| 3 | 13 | 0.1 | 0.1831 | 0.3651 | 0.1222 |
| 4 | 8 | 100.0 | 0.1827 | 0.4784 | 0.1129 |
| 5 | 15 | 100.0 | 0.1822 | 0.2893 | 0.1330 |

### n=5

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 13 | 10.0 | 0.3246 | 0.4156 | 0.2663 |
| 2 | 29 | 0.01 | 0.3183 | 0.3644 | 0.2826 |
| 3 | 27 | 0.5 | 0.2984 | 0.2630 | 0.3447 |
| 4 | 30 | 0.05 | 0.2862 | 0.2955 | 0.2774 |
| 5 | 20 | 100.0 | 0.2807 | 0.3279 | 0.2453 |

### n=10

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 26 | 0.01 | 0.3500 | 0.3220 | 0.3834 |
| 2 | 14 | 5.0 | 0.3433 | 0.3297 | 0.3581 |
| 3 | 11 | 10.0 | 0.3238 | 0.3397 | 0.3093 |
| 4 | 29 | 0.5 | 0.3229 | 0.3630 | 0.2908 |
| 5 | 17 | 0.05 | 0.3216 | 0.2995 | 0.3472 |

### n=20

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 28 | 0.1 | 0.4423 | 0.4534 | 0.4317 |
| 2 | 26 | 50.0 | 0.4296 | 0.4277 | 0.4316 |
| 3 | 18 | 100.0 | 0.4281 | 0.4038 | 0.4556 |
| 4 | 14 | 0.01 | 0.4214 | 0.4212 | 0.4216 |
| 5 | 29 | 50.0 | 0.4191 | 0.4025 | 0.4372 |
