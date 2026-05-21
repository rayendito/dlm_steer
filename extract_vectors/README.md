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
python extract_vectors/extract_steer_vectors.py --data imdb --num-samples 20
python extract_vectors/extract_steer_vectors.py --data imdb --num-samples 0   # love / hate
python extract_vectors/extract_steer_vectors.py --data cats-dogs --num-samples 100
python extract_vectors/extract_steer_vectors.py --data cats-dogs --num-samples 0   # cat / dog
```

- `--data imdb` (default): `--num-samples 0` → **love** / **hate** → `steer_vectors/diffusion-imdb-n0.pt`; `N > 0` → first `N` rows from `benchmarks/imdb/val_pos.csv` and `val_neg.csv` → `steer_vectors/diffusion-imdb-n{N}.pt`.
- `--data cats-dogs`: **cat** = positive, **dog** = negative; `N > 0` → first `N` of each from `benchmarks/cats_dogs/train.csv` (up to ~1000 per class); `N = 0` → single tokens **cat** / **dog** → `steer_vectors/diffusion-catdog-n0.pt`. Val split (`val.csv`, ~100 cat / ~100 dog) is for downstream eval, not extraction.

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

Sweeps **α × layer** for a chosen vector file so you can pick **α** and **layer** before larger experiments. Outputs: **`extract_vectors/results/results_{abbrev}_{tag}/`** (`scores.json`, `eval_scores.json`, heatmaps). `{abbrev}` is `pos`/`neg` (IMDB) or `cats`/`dogs` (cat–dog). `{tag}` comes from the vector filename (e.g. `diffusion-imdb-n20.pt` → `20`). **`--data`** defaults from **`--vectors`** (`diffusion-imdb-*` → imdb, `diffusion-catdog-*` → cats-dogs). Current α grid: **10–100 step 5** (see `ALPHAS` in `resteer_val_sweep_eval.py`).

**Examples:**

```bash
python extract_vectors/resteer_val_sweep_eval.py --direction negative --vectors steer_vectors/diffusion-imdb-n20.pt
python extract_vectors/resteer_val_sweep_eval.py --direction cats --vectors steer_vectors/diffusion-catdog-n100.pt
python extract_vectors/resteer_val_sweep_eval.py --direction dogs --vectors steer_vectors/diffusion-catdog-n100.pt
```

**Directions:** IMDB → `positive` / `negative` (prompts from opposite val split). cats-dogs → `cats` / `dogs` (aliases `positive`/`negative`; val prompts from opposite concept in `benchmarks/cats_dogs/val.csv`).

**Evaluation:** `utils/eval_utils.py` (Qwen2.5-0.5B-Instruct; labels `positive`/`negative` or `cat`/`dog` matching the vector file).

**Steering:** the sweep calls **`resteer_v2`** in `llada/generate.py` (identify → mask → `refill_steps` steered refills per outer `resteer_steps` iteration). **`RESTEER_STEPS`=1**, **`REFILL_STEPS`=1**, **`PERPLEXITY_THRESHOLD`=10000**, **`SENTIMENT_THRESHOLD`=0.1**, and related knobs live at the top of `resteer_val_sweep_eval.py`.

## 4. Merge pos + neg summaries (`extract_vectors/merge_eval_results.py`)

After you have **paired** folders under **`extract_vectors/results/`**, run the merge script. It pairs:

- **IMDB:** `results_pos_{tag}` + `results_neg_{tag}`
- **cats-dogs:** `results_cats_{tag}` + `results_dogs_{tag}`

(same `tag`, both with `eval_scores.json`). It builds per-direction top‑5, **cross-direction** top‑5 on the full layer×α union grid, and heatmaps **`avg_combined_imdb_{tag}.png`** / **`avg_combined_catdog_{tag}.png`**. Output: **`extract_vectors/results.json`** (`per_tag` keyed by `merge_key`: `{tag}` for IMDB, `catdog-{tag}` for cats-dogs).

**Example:**

```bash
python extract_vectors/merge_eval_results.py
```

---

## Top 5 by vector variation (`n`)

Snapshot from **`extract_vectors/results.json`** (eval via `utils/eval_utils.py`; α grid 10–100 step 5). **cross_hm** = harmonic mean of both direction scores at the same `(layer, α)`; table = top 5 over the full union grid with **unique layers** (greedy by `cross_hm`). Heatmaps: **`avg_combined_imdb_{n}.png`** / **`avg_combined_catdog_{n}.png`**.

### IMDB (`diffusion-imdb-n*`, tags `0`–`100`)

#### n=0

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 11 | 45 | 0.2368 | 0.1997 | 0.2910 |
| 2 | 32 | 90 | 0.2278 | 0.2176 | 0.2390 |
| 3 | 0 | 20 | 0.2240 | 0.1638 | 0.3540 |
| 4 | 23 | 10 | 0.2148 | 0.1695 | 0.2934 |
| 5 | 4 | 10 | 0.2137 | 0.1693 | 0.2898 |

#### n=1

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 9 | 15 | 0.3545 | 0.2825 | 0.4757 |
| 2 | 11 | 15 | 0.3308 | 0.3435 | 0.3191 |
| 3 | 4 | 25 | 0.3147 | 0.2284 | 0.5060 |
| 4 | 7 | 20 | 0.3067 | 0.2217 | 0.4973 |
| 5 | 5 | 20 | 0.2993 | 0.2397 | 0.3985 |

#### n=5

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15 | 0.3310 | 0.2716 | 0.4236 |
| 2 | 11 | 30 | 0.2655 | 0.2199 | 0.3350 |
| 3 | 5 | 75 | 0.2533 | 0.2057 | 0.3297 |
| 4 | 6 | 60 | 0.2429 | 0.1829 | 0.3616 |
| 5 | 10 | 30 | 0.2324 | 0.1886 | 0.3027 |

#### n=10

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15 | 0.3729 | 0.3165 | 0.4537 |
| 2 | 6 | 40 | 0.2904 | 0.2279 | 0.4002 |
| 3 | 5 | 55 | 0.2853 | 0.2556 | 0.3227 |
| 4 | 2 | 90 | 0.2852 | 0.2406 | 0.3500 |
| 5 | 7 | 55 | 0.2820 | 0.2116 | 0.4225 |

#### n=20

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15 | 0.4725 | 0.3832 | 0.6160 |
| 2 | 10 | 55 | 0.3118 | 0.2605 | 0.3884 |
| 3 | 8 | 85 | 0.2948 | 0.2384 | 0.3860 |
| 4 | 5 | 100 | 0.2919 | 0.2261 | 0.4116 |
| 5 | 11 | 50 | 0.2880 | 0.2129 | 0.4449 |

#### n=50

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 15 | 0.5449 | 0.5200 | 0.5724 |
| 2 | 29 | 30 | 0.3026 | 0.2218 | 0.4756 |
| 3 | 1 | 70 | 0.2958 | 0.2389 | 0.3883 |
| 4 | 11 | 70 | 0.2834 | 0.2119 | 0.4276 |
| 5 | 27 | 30 | 0.2800 | 0.2218 | 0.3793 |

#### n=100

| rank | layer | α | cross_hm | pos_hm | neg_hm |
|------|-------|---|----------|--------|--------|
| 1 | 32 | 25 | 0.4754 | 0.3839 | 0.6241 |
| 2 | 25 | 25 | 0.3114 | 0.2252 | 0.5044 |
| 3 | 30 | 30 | 0.2985 | 0.2058 | 0.5436 |
| 4 | 29 | 30 | 0.2978 | 0.2237 | 0.4454 |
| 5 | 28 | 35 | 0.2849 | 0.2038 | 0.4730 |

### cats-dogs (`diffusion-catdog-n*`, tags `0`–`100`)

#### n=0

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 6 | 10 | 0.1842 | 0.1479 | 0.2443 |
| 2 | 7 | 10 | 0.1829 | 0.1484 | 0.2383 |
| 3 | 8 | 15 | 0.1747 | 0.1453 | 0.2189 |
| 4 | 5 | 10 | 0.1719 | 0.1280 | 0.2616 |
| 5 | 15 | 10 | 0.1699 | 0.1504 | 0.1952 |

#### n=1

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 2 | 40 | 0.2142 | 0.1925 | 0.2414 |
| 2 | 29 | 65 | 0.2103 | 0.2290 | 0.1945 |
| 3 | 31 | 25 | 0.2100 | 0.1967 | 0.2251 |
| 4 | 30 | 20 | 0.2074 | 0.1750 | 0.2546 |
| 5 | 3 | 35 | 0.1987 | 0.1831 | 0.2172 |

#### n=5

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 30 | 20 | 0.2620 | 0.3297 | 0.2173 |
| 2 | 31 | 75 | 0.2599 | 0.3247 | 0.2167 |
| 3 | 29 | 10 | 0.2435 | 0.3093 | 0.2008 |
| 4 | 28 | 15 | 0.2286 | 0.2754 | 0.1954 |
| 5 | 27 | 55 | 0.2194 | 0.2502 | 0.1953 |

#### n=10

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 31 | 75 | 0.3139 | 0.3518 | 0.2833 |
| 2 | 30 | 100 | 0.2879 | 0.2946 | 0.2816 |
| 3 | 29 | 35 | 0.2480 | 0.2201 | 0.2839 |
| 4 | 27 | 20 | 0.2462 | 0.2732 | 0.2241 |
| 5 | 28 | 15 | 0.2411 | 0.2649 | 0.2211 |

#### n=20

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 31 | 75 | 0.3070 | 0.2798 | 0.3402 |
| 2 | 30 | 100 | 0.2917 | 0.2370 | 0.3793 |
| 3 | 28 | 80 | 0.2463 | 0.2629 | 0.2316 |
| 4 | 27 | 95 | 0.2413 | 0.2125 | 0.2791 |
| 5 | 29 | 75 | 0.2330 | 0.1740 | 0.3525 |

#### n=50

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 30 | 80 | 0.2539 | 0.2155 | 0.3088 |
| 2 | 31 | 45 | 0.2524 | 0.2253 | 0.2871 |
| 3 | 28 | 15 | 0.2288 | 0.2715 | 0.1977 |
| 4 | 32 | 95 | 0.2153 | 0.2755 | 0.1766 |
| 5 | 29 | 15 | 0.2095 | 0.2429 | 0.1841 |

#### n=100

| rank | layer | α | cross_hm | cats_hm | dogs_hm |
|------|-------|---|----------|--------|--------|
| 1 | 30 | 15 | 0.2511 | 0.2948 | 0.2187 |
| 2 | 28 | 20 | 0.2453 | 0.2821 | 0.2170 |
| 3 | 31 | 85 | 0.2405 | 0.2216 | 0.2628 |
| 4 | 29 | 15 | 0.2197 | 0.2539 | 0.1936 |
| 5 | 32 | 95 | 0.2184 | 0.2875 | 0.1761 |

