# Extract vectors pipeline

## 1. Benchmarks (`extract_vectors/download_dataset.py`)

Downloads **`stanfordnlp/imdb`** (via Hugging Face `datasets`) and writes balanced CSVs:

- **Train:** 2,000 positive + 2,000 negative (random per label; `--seed` configurable).
- **Val:** 2,000 positive + 2,000 negative from the **test** split (we use test as val).

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

Sweeps **α × layer** for a chosen vector file so you can pick **α** and **layer** before larger experiments. Outputs: **`extract_vectors/results/results_{pos|neg}_{tag}/`** (`scores.json`, `eval_scores.json`, per-direction heatmaps). `{pos|neg}` comes from `--direction` (`positive` → `pos`, `negative` → `neg`). `{tag}` is inferred from the vector filename (e.g. `diffusion-val-n20.pt` → `20` → `results/results_neg_20/` with `--direction negative`). Current α grid: **10–100 step 5** (see `ALPHAS` in `resteer_val_sweep_eval.py`).

**Example:**

```bash
python extract_vectors/resteer_val_sweep_eval.py --direction negative --vectors steer_vectors/diffusion-val-n20.pt
```

**Evaluation:** sentiment and perplexity use `eval_dito.py` (same metrics as elsewhere in the repo).

**Steering:** the sweep calls **`resteer_v2`** in `llada/generate.py` (identify → mask → `refill_steps` steered refills per outer `resteer_steps` iteration). **`RESTEER_STEPS`=1**, **`REFILL_STEPS`=1**, **`PERPLEXITY_THRESHOLD`=10000**, **`SENTIMENT_THRESHOLD`=0.1**, and related knobs live at the top of `resteer_val_sweep_eval.py`.

## 4. Merge pos + neg summaries (`extract_vectors/merge_eval_results.py`)

After you have **paired** folders under **`extract_vectors/results/`** — `results_pos_{tag}/` and `results_neg_{tag}/` (same `tag`, e.g. `0`, `20`, `200`) — run the merge script. It auto-detects which tags have both directions, reads each side’s `eval_scores.json`, copies per-direction top‑5 harmonic-mean rows (unique layer), pairs those ranks, and builds **cross-direction** top‑5 over the **full** layer×α union grid (missing direction scores padded with 0), again **one α per layer** chosen by greedy cross harmonic mean. It also writes **`extract_vectors/results/heatmaps/avg_combined_{tag}.png`** (pos / neg / cross harmonic panels).

**Example:**

```bash
python extract_vectors/merge_eval_results.py
```

---

## Top 5 by vector variation (`n`)

Snapshot from **`extract_vectors/results.json`** (paired tags: `0`, `1`, `5`, `10`, `20`, `50`, `100`, `200`, `500`, `1000`, `2000`; α grid 10–100 step 5). Per-direction top‑5: unique layers; harmonic mean = sentiment vs. 1 − robust-normalized perplexity, as in the sweep. **cross_hm** = harmonic mean of pos- and neg-direction harmonic scores on the **same** `(layer, α)`; the cross table is **top 5 over the full union grid with unique layers** (greedy by `cross_hm`). Heatmaps: **`extract_vectors/results/heatmaps/avg_combined_{n}.png`**.

### n=0

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 11 | 45.0 | 0.2368 | 0.1997 | 0.2910 |
| 2 | 32 | 90.0 | 0.2278 | 0.2176 | 0.2390 |
| 3 | 0 | 20.0 | 0.2240 | 0.1638 | 0.3540 |
| 4 | 23 | 10.0 | 0.2148 | 0.1695 | 0.2934 |
| 5 | 4 | 10.0 | 0.2137 | 0.1693 | 0.2898 |

### n=1

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 9 | 15.0 | 0.3545 | 0.2825 | 0.4757 |
| 2 | 11 | 15.0 | 0.3308 | 0.3435 | 0.3191 |
| 3 | 4 | 25.0 | 0.3147 | 0.2284 | 0.5060 |
| 4 | 7 | 20.0 | 0.3067 | 0.2217 | 0.4973 |
| 5 | 5 | 20.0 | 0.2993 | 0.2397 | 0.3985 |

### n=5

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15.0 | 0.3310 | 0.2716 | 0.4236 |
| 2 | 11 | 30.0 | 0.2655 | 0.2199 | 0.3350 |
| 3 | 5 | 75.0 | 0.2533 | 0.2057 | 0.3297 |
| 4 | 6 | 60.0 | 0.2429 | 0.1829 | 0.3616 |
| 5 | 10 | 30.0 | 0.2324 | 0.1886 | 0.3027 |

### n=10

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15.0 | 0.3729 | 0.3165 | 0.4537 |
| 2 | 6 | 40.0 | 0.2904 | 0.2279 | 0.4002 |
| 3 | 5 | 55.0 | 0.2853 | 0.2556 | 0.3227 |
| 4 | 2 | 90.0 | 0.2852 | 0.2406 | 0.3500 |
| 5 | 7 | 55.0 | 0.2820 | 0.2116 | 0.4225 |

### n=20

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15.0 | 0.4725 | 0.3832 | 0.6160 |
| 2 | 10 | 55.0 | 0.3118 | 0.2605 | 0.3884 |
| 3 | 8 | 85.0 | 0.2948 | 0.2384 | 0.3860 |
| 4 | 5 | 100.0 | 0.2919 | 0.2261 | 0.4116 |
| 5 | 11 | 50.0 | 0.2880 | 0.2129 | 0.4449 |

### n=50

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15.0 | 0.5449 | 0.5200 | 0.5724 |
| 2 | 29 | 30.0 | 0.3026 | 0.2218 | 0.4756 |
| 3 | 1 | 70.0 | 0.2958 | 0.2389 | 0.3883 |
| 4 | 11 | 70.0 | 0.2834 | 0.2119 | 0.4276 |
| 5 | 27 | 30.0 | 0.2800 | 0.2218 | 0.3793 |

### n=100

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 25.0 | 0.4754 | 0.3839 | 0.6241 |
| 2 | 25 | 25.0 | 0.3114 | 0.2252 | 0.5044 |
| 3 | 30 | 30.0 | 0.2985 | 0.2058 | 0.5436 |
| 4 | 29 | 30.0 | 0.2978 | 0.2237 | 0.4454 |
| 5 | 28 | 35.0 | 0.2849 | 0.2038 | 0.4730 |

### n=200

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 25.0 | 0.4953 | 0.4020 | 0.6450 |
| 2 | 28 | 10.0 | 0.3515 | 0.2616 | 0.5352 |
| 3 | 25 | 25.0 | 0.3248 | 0.2547 | 0.4481 |
| 4 | 29 | 30.0 | 0.3193 | 0.2432 | 0.4645 |
| 5 | 31 | 35.0 | 0.2877 | 0.1965 | 0.5366 |

### n=500

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 25.0 | 0.4628 | 0.3613 | 0.6435 |
| 2 | 29 | 30.0 | 0.3342 | 0.2546 | 0.4862 |
| 3 | 25 | 25.0 | 0.3313 | 0.2370 | 0.5499 |
| 4 | 28 | 35.0 | 0.2992 | 0.2211 | 0.4628 |
| 5 | 27 | 30.0 | 0.2751 | 0.2216 | 0.3629 |

### n=1000

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15.0 | 0.4887 | 0.4023 | 0.6222 |
| 2 | 29 | 30.0 | 0.3527 | 0.2748 | 0.4921 |
| 3 | 25 | 30.0 | 0.3296 | 0.2522 | 0.4755 |
| 4 | 27 | 35.0 | 0.3049 | 0.2430 | 0.4092 |
| 5 | 30 | 40.0 | 0.3005 | 0.2167 | 0.4902 |

### n=2000

**Pos + Neg (layer, α): top 5 by cross-direction harmonic mean**

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 25.0 | 0.4825 | 0.3876 | 0.6390 |
| 2 | 29 | 30.0 | 0.3795 | 0.2970 | 0.5252 |
| 3 | 25 | 25.0 | 0.3580 | 0.2696 | 0.5325 |
| 4 | 30 | 40.0 | 0.3033 | 0.2108 | 0.5407 |
| 5 | 26 | 15.0 | 0.2966 | 0.2531 | 0.3582 |
