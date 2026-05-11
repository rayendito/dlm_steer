#!/usr/bin/env python3
"""
Val steering sweep + eval: ``resteer_v2_val`` on full val texts (no prefix + ``generate`` tail).

Each CSV row is treated as the **complete** sequence to steer in place: tokenize (truncate
if needed), run ``resteer_v2_val`` (identify tokens → mask → one steered forward, no refill
loop), decode.

Same vector construction as sandbox.py (L2 norm, neg−pos; ``--direction positive`` flips).

Run from repo root:
  uv run python extract_vectors/resteer_val_sweep_eval.py \\
    --direction negative --vectors steer_vectors/diffusion-val_extract_n20.pt

Tweak sweep / paths / resteer hyperparameters below (no CLI besides direction, vectors, --skip-eval).

Prompt source (opposite-class reviews): ``--direction positive`` uses **val_neg** only
(steer negative-review text toward positive). ``--direction negative`` uses **val_pos** only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Script is run as a file (`python extract_vectors/...`); repo root must be on path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_root = str(_REPO_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from eval_dito import perplexity as dito_perplexity
from eval_dito import score_labels
from llada.configuration_llada import LLaDAConfig
from llada.generate import resteer_v2_val
from llada.modeling_llada import LLaDAModelLM

# --- hardcoded sweep / IO -------------------------------------------------
VAL_POS = Path("benchmarks/val_pos.csv")
VAL_NEG = Path("benchmarks/val_neg.csv")
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda"
SEED = 42

ALPHAS = [
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    0.0001, 0.001, 0.01, 0.1, 1, 10, 100
]
LAYER_MIN = 0
LAYER_MAX = 33
# resteer_v2_val: one identify→mask→steered forward (see llada.generate.resteer_v2_val)
MAX_STEER_SEQ_LEN = 1024
IDENTIFY_TEMPERATURE = 0.0001

# Writes under ``extract_vectors/results_{tag}/`` where ``tag`` is inferred from ``--vectors``
# (e.g. ``..._n20.pt`` → ``results_20``, ``..._love_hate.pt`` or ``num-samples 0`` naming → ``results_0``).
RESULTS_PARENT = Path("extract_vectors")
EVAL_BATCH_SIZE = 4
# Mini-batch of full reviews per resteer call (same α × layer for all rows in batch).
RESTEER_BATCH_SIZE = 1


def results_tag_from_vectors_path(vectors_path: Path) -> str:
    """Short suffix for output folder, aligned with extract_steer_vectors filenames."""
    stem = vectors_path.stem.lower()
    if "love_hate" in stem:
        return "0"
    for pat in (r"_n(\d+)$", r"-n(\d+)$", r"_n(\d+)", r"-n(\d+)"):
        m = re.search(pat, stem)
        if m:
            return m.group(1)
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:48]
    return slug if slug else "unknown"


def resolve_result_paths(vectors_path: Path) -> tuple[Path, Path, Path, Path, str]:
    tag = results_tag_from_vectors_path(vectors_path)
    out_dir = RESULTS_PARENT / f"results_{tag}"
    return (
        out_dir / "scores.json",
        out_dir / "eval_scores.json",
        out_dir / "heatmaps.png",
        out_dir / "heatmaps_avg.png",
        tag,
    )


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def load_steering_prompt_texts(
    val_pos: Path,
    val_neg: Path,
    steering_direction: str,
) -> tuple[list[tuple[int, str, str]], str]:
    """
    Load prefix texts from the class *opposite* the steering target:

    - ``steering_direction == "positive"`` → use **negative** reviews (`val_neg`).
    - ``steering_direction == "negative"`` → use **positive** reviews (`val_pos`).

    Returns (rows, description) where rows are (prompt_idx, text, split_label).
    """
    if steering_direction == "positive":
        label, path = "neg", val_neg
        desc = f"val_neg only (negative-review prefixes; steer toward positive)"
    elif steering_direction == "negative":
        label, path = "pos", val_pos
        desc = f"val_pos only (positive-review prefixes; steer toward negative)"
    else:
        raise ValueError("steering_direction must be positive or negative")

    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")

    rows: list[tuple[int, str, str]] = []
    idx = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"{path} must have a 'text' column")
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                rows.append((idx, t, label))
                idx += 1
    if not rows:
        raise ValueError(f"No texts loaded from {path}")

    ## DEBUG
    text = "This movie is sucks so bad, action is poor, and the plot is stupid."
    rows = [(0, text, label)]
    # rows = rows[:1]

    return rows, desc


def build_steer_bases(
    vectors_path: Path,
    direction: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], int]:
    if not vectors_path.is_file():
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")
    sentiment_vectors = torch.load(vectors_path, map_location=device)
    pos_vectors = sentiment_vectors["positive"]
    neg_vectors = sentiment_vectors["negative"]
    steer_all = tuple(
        l2_normalize(neg_vectors[i].to(dtype=dtype, device=device))
        - l2_normalize(pos_vectors[i].to(dtype=dtype, device=device))
        for i in range(len(pos_vectors))
    )
    if direction == "positive":
        steer_all = tuple(-v for v in steer_all)
    elif direction != "negative":
        raise ValueError("--direction must be positive or negative")
    return steer_all, len(steer_all)


def parse_sweep_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Invalid sweep JSON: 'results' must be a list.")
    return [
        r
        for r in results
        if isinstance(r, dict)
        and r.get("mode") == "steered"
        and isinstance(r.get("layer"), int)
        and r["layer"] >= 0
    ]


def run_evaluation(
    sweep_file: Path,
    direction: str,
    *,
    batch_size: int,
    eval_out_json: Path,
    out_png: Path,
    out_png_avg: Path,
) -> None:
    sentiment_key = (
        "negative_sentiment_prob"
        if direction == "negative"
        else "positive_sentiment_prob"
    )
    sentiment_title = (
        "Negative sentiment P(token)"
        if direction == "negative"
        else "Positive sentiment P(token)"
    )

    with sweep_file.open(encoding="utf-8") as f:
        payload = json.load(f)

    rows = parse_sweep_results(payload)
    if not rows:
        print("No steered rows in sweep JSON.", file=sys.stderr)
        sys.exit(1)

    per_sample_scores: dict[tuple[int, int, float], dict[str, float]] = {}

    total = len(rows)
    n_batches = (total + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, total, batch_size),
        desc="eval_dito batches",
        total=n_batches,
        file=sys.stderr,
    ):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        texts = [str(r.get("generation", "")) for r in batch]
        _, p_pos, p_neg = score_labels(texts)
        ppls = dito_perplexity(texts)
        tgt = p_neg if direction == "negative" else p_pos

        for i, row in enumerate(batch):
            prompt_idx = int(row.get("prompt_idx", -1))
            layer = int(row["layer"])
            alpha = float(row["alpha"])
            s = {
                sentiment_key: float(tgt[i].item()),
                "perplexity": float(ppls[i].item()),
            }
            per_sample_scores[(prompt_idx, layer, alpha)] = s

    finite_ppls = [
        v["perplexity"]
        for v in per_sample_scores.values()
        if np.isfinite(v["perplexity"])
    ]
    max_finite_ppl = max(finite_ppls) if finite_ppls else 1.0
    for v in per_sample_scores.values():
        ppl = v["perplexity"]
        if not np.isfinite(ppl):
            v["perplexity"] = max_finite_ppl

    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]] = {}
    for (prompt_idx, layer, alpha), s in per_sample_scores.items():
        pm = per_prompt_scores.setdefault(prompt_idx, {})
        pm[(layer, alpha)] = s

    out_payload = {
        "sweep_file": str(sweep_file),
        "direction": direction,
        "scorer": "eval_dito.score_labels + eval_dito.perplexity",
        "sentiment_metric": sentiment_key,
        "results_per_sample": [
            {
                "prompt_idx": p,
                "layer": l,
                "alpha": a,
                sentiment_key: per_sample_scores[(p, l, a)][sentiment_key],
                "perplexity": per_sample_scores[(p, l, a)]["perplexity"],
            }
            for (p, l, a) in sorted(per_sample_scores.keys())
        ],
    }
    eval_out_json.parent.mkdir(parents=True, exist_ok=True)
    with eval_out_json.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Wrote {eval_out_json}", file=sys.stderr)

    _plot_heatmaps(per_prompt_scores, sentiment_key, sentiment_title, out_png)
    avg_scores = _build_avg(per_prompt_scores, sentiment_key)
    _print_top5(avg_scores, sentiment_key)
    _plot_avg(avg_scores, sentiment_key, sentiment_title, out_png_avg)


def _build_avg(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]],
    sentiment_key: str,
) -> dict[tuple[int, float], dict[str, float]]:
    from collections import defaultdict

    agg: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(
        lambda: {sentiment_key: [], "perplexity": []}
    )
    for pmap in per_prompt_scores.values():
        for key, s in pmap.items():
            agg[key][sentiment_key].append(float(s[sentiment_key]))
            agg[key]["perplexity"].append(float(s["perplexity"]))

    avg: dict[tuple[int, float], dict[str, float]] = {}
    for key, vals in agg.items():
        avg[key] = {
            sentiment_key: sum(vals[sentiment_key]) / max(1, len(vals[sentiment_key])),
            "perplexity": sum(vals["perplexity"]) / max(1, len(vals["perplexity"])),
        }
    return avg


def _matrices_from_scores(
    scores: dict[tuple[int, float], dict[str, float]],
    sentiment_key: str,
):
    if not scores:
        return [], [], np.array([]), np.array([]), np.array([])
    layers = sorted({k[0] for k in scores.keys()})
    alphas = sorted({k[1] for k in scores.keys()})
    li_map = {l: i for i, l in enumerate(layers)}
    ai_map = {a: i for i, a in enumerate(alphas)}
    M_sent = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    M_ppl = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    for (layer, alpha), s in scores.items():
        M_sent[li_map[layer], ai_map[alpha]] = float(s[sentiment_key])
        M_ppl[li_map[layer], ai_map[alpha]] = float(s["perplexity"])
    # fifth slot kept for API parity with empty branch (callers unpack with _)
    M_mul = M_sent * M_ppl
    return layers, alphas, M_sent, M_ppl, M_mul


def _robust_normalize(M: Any, lo_pct: float = 0.0, hi_pct: float = 85.0) -> Any:
    finite = np.isfinite(M)
    if not np.any(finite):
        return np.zeros_like(M)
    vals = M[finite]
    lo = float(np.nanpercentile(vals, lo_pct))
    hi = float(np.nanpercentile(vals, hi_pct))
    if hi < lo:
        lo, hi = hi, lo
    clipped = np.clip(M, lo, hi)
    cmin = float(np.nanmin(clipped[finite]))
    cmax = float(np.nanmax(clipped[finite]))
    if cmax > cmin:
        return (clipped - cmin) / (cmax - cmin)
    return np.zeros_like(M)


def _fmt_alpha(a: float) -> str:
    if abs(a - round(a)) < 1e-9:
        return str(int(round(a)))
    return f"{a:.3g}"


def _draw_three(
    axes: list[Any],
    layers: list[int],
    alphas: list[float],
    M_sent: Any,
    M_ppl_raw: Any,
    row_title: str,
    sentiment_title: str,
) -> None:
    M_ppl_plot = _robust_normalize(M_ppl_raw)
    M_combo = M_sent * (1.0 - M_ppl_plot)
    fig = axes[0].figure
    for ax, M, title, cmap, vmin, vmax, cbl in [
        (
            axes[0],
            M_sent,
            f"{row_title} - {sentiment_title}",
            "magma",
            0.0,
            1.0,
            "probability",
        ),
        (
            axes[1],
            M_ppl_plot,
            f"{row_title} - Perplexity (normalized)",
            "viridis_r",
            0.0,
            1.0,
            "normalized ppl",
        ),
        (
            axes[2],
            M_combo,
            f"{row_title} - Combined (higher is better)",
            "cividis",
            0.0,
            1.0,
            "score",
        ),
    ]:
        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("layer")
        ax.set_title(title, fontsize=10)
        if alphas:
            step_x = max(1, len(alphas) // 8)
            xt = list(range(0, len(alphas), step_x))
            if (len(alphas) - 1) not in xt:
                xt.append(len(alphas) - 1)
            ax.set_xticks(xt)
            ax.set_xticklabels([_fmt_alpha(alphas[i]) for i in xt], rotation=30, ha="right")
        if layers:
            step_y = max(1, len(layers) // 16)
            yt = list(range(0, len(layers), step_y))
            ax.set_yticks(yt)
            ax.set_yticklabels([str(layers[i]) for i in yt])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbl)


def _plot_heatmaps(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]],
    sentiment_key: str,
    sentiment_title: str,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_ids = sorted(per_prompt_scores.keys())
    n_rows = max(1, len(prompt_ids))
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    for i, pid in enumerate(prompt_ids):
        layers, alphas, M_s, M_p, _ = _matrices_from_scores(
            per_prompt_scores[pid], sentiment_key
        )
        _draw_three(
            list(axes[i]), layers, alphas, M_s, M_p, f"Prompt {pid}", sentiment_title
        )
    fig.suptitle(
        "Val resteer_v2_val sweep: sentiment (direction-aligned), perplexity, combined",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {out_png}", file=sys.stderr)


def _plot_avg(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    sentiment_key: str,
    sentiment_title: str,
    out_png_avg: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, alphas, M_s, M_p, _ = _matrices_from_scores(avg_scores, sentiment_key)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    _draw_three(list(axes), layers, alphas, M_s, M_p, "Average (all prompts)", sentiment_title)
    fig.suptitle("Val resteer_v2_val sweep averaged across prompts", fontsize=12)
    out_png_avg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_avg, dpi=150)
    plt.close(fig)
    print(f"Saved avg heatmap: {out_png_avg}", file=sys.stderr)


def _print_top5(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    sentiment_key: str,
) -> None:
    if not avg_scores:
        print("No averaged scores for top-5.", file=sys.stderr)
        return
    cells = list(avg_scores.items())

    by_sent = sorted(
        cells,
        key=lambda kv: float(kv[1][sentiment_key]),
        reverse=True,
    )[:5]
    print(
        f"\nTop 5 averaged combinations by highest {sentiment_key} (combination-wise):",
        file=sys.stderr,
    )
    for rank, ((layer, alpha), s) in enumerate(by_sent, start=1):
        sent = float(s[sentiment_key])
        ppl = float(s["perplexity"])
        print(
            f"{rank}. layer={layer}, alpha={alpha:g}, {sentiment_key}={sent:.4f}, ppl={ppl:.4f}",
            file=sys.stderr,
        )

    def _ppl_sort_key(kv: tuple[tuple[int, float], dict[str, float]]) -> float:
        p = float(kv[1]["perplexity"])
        return p if np.isfinite(p) else float("inf")

    by_ppl = sorted(cells, key=_ppl_sort_key)[:5]
    print(
        "\nTop 5 averaged combinations by lowest perplexity (combination-wise):",
        file=sys.stderr,
    )
    for rank, ((layer, alpha), s) in enumerate(by_ppl, start=1):
        sent = float(s[sentiment_key])
        ppl = float(s["perplexity"])
        print(
            f"{rank}. layer={layer}, alpha={alpha:g}, ppl={ppl:.4f}, {sentiment_key}={sent:.4f}",
            file=sys.stderr,
        )

    ppl_vals = np.array([float(v["perplexity"]) for _, v in cells], dtype=float)
    finite = np.isfinite(ppl_vals)
    ppl_norm = np.zeros_like(ppl_vals, dtype=float)
    if np.any(finite):
        vals = ppl_vals[finite]
        lo = float(np.nanpercentile(vals, 0))
        hi = float(np.nanpercentile(vals, 85))
        if hi < lo:
            lo, hi = hi, lo
        clipped = np.clip(ppl_vals, lo, hi)
        cmin = float(np.nanmin(clipped[finite]))
        cmax = float(np.nanmax(clipped[finite]))
        if cmax > cmin:
            ppl_norm = (clipped - cmin) / (cmax - cmin)

    ranked = []
    for i, ((layer, alpha), s) in enumerate(cells):
        sent = float(s[sentiment_key])
        ppl = float(s["perplexity"])
        combo = sent * (1.0 - float(ppl_norm[i]))
        ranked.append((combo, layer, alpha, sent, ppl))
    ranked.sort(key=lambda x: x[0], reverse=True)

    topk: list[Any] = []
    used_layers: set[int] = set()
    for item in ranked:
        _, layer, _, _, _ = item
        if layer in used_layers:
            continue
        topk.append(item)
        used_layers.add(layer)
        if len(topk) == 5:
            break

    print(
        "\nTop 5 averaged combinations by combined score "
        "(unique layers, higher is better):",
        file=sys.stderr,
    )
    for rank, (combo, layer, alpha, sent, ppl) in enumerate(topk, start=1):
        print(
            f"{rank}. layer={layer}, alpha={alpha:g}, combined={combo:.4f}, "
            f"{sentiment_key}={sent:.4f}, ppl={ppl:.4f}",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Val steering sweep (LLaDA resteer_v2_val on full val texts). "
        "Edit module-level constants for paths, alphas, layers, resteer hyperparameters, outputs."
    )
    ap.add_argument(
        "--direction",
        choices=["positive", "negative"],
        required=True,
    )
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    (
        out_sweep_json,
        out_eval_json,
        out_png,
        out_png_avg,
        vectors_results_tag,
    ) = resolve_result_paths(Path(args.vectors))
    print(
        f"Outputs → {RESULTS_PARENT / f'results_{vectors_results_tag}'} "
        f"(tag from vectors: {vectors_results_tag})",
        flush=True,
    )

    device = DEVICE
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    val_rows, prompt_desc = load_steering_prompt_texts(
        VAL_POS, VAL_NEG, args.direction
    )
    texts = [t for _, t, _ in val_rows]
    prompt_idxs = [i for i, _, _ in val_rows]
    print(f"Sweep prompts: {len(val_rows)} rows — {prompt_desc}", flush=True)

    alphas = list(ALPHAS)
    steer_bases, num_layers = build_steer_bases(
        args.vectors, args.direction, device, torch.bfloat16
    )

    layer_lo = max(0, LAYER_MIN)
    layer_hi = min(LAYER_MAX, num_layers - 1)
    if layer_lo > layer_hi:
        raise SystemExit(
            f"Invalid layer range {LAYER_MIN}-{LAYER_MAX} (model has {num_layers} layers)"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config = LLaDAConfig.from_pretrained(MODEL_ID)
    model = LLaDAModelLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    results: list[dict[str, Any]] = []

    layer_range = range(layer_lo, layer_hi + 1)
    sweep_pairs = list(product(alphas, layer_range))
    for alpha, layer in tqdm(
        sweep_pairs,
        desc="resteer_v2_val sweep (α × layer)",
        unit="pair",
    ):
        steer_vectors = {layer: float(alpha) * steer_bases[layer]}
        gb = max(1, RESTEER_BATCH_SIZE)
        n_txt = len(texts)
        n_chunks = (n_txt + gb - 1) // gb
        for c in tqdm(
            range(0, n_txt, gb),
            desc=f"α={alpha:g} L={layer}",
            total=n_chunks,
            leave=False,
        ):
            chunk_pidx = prompt_idxs[c : c + gb]
            chunk_texts = texts[c : c + gb]
            tokenized_inputs = tokenizer(
                chunk_texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=MAX_STEER_SEQ_LEN,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                steered_ids = resteer_v2_val(
                    model,
                    tokenized_inputs,
                    steer_vectors,
                    identify_temperature=IDENTIFY_TEMPERATURE,
                )
            decoded = tokenizer.batch_decode(
                steered_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for pidx, text, steered_text in zip(
                chunk_pidx,
                chunk_texts,
                decoded,
                strict=True,
            ):
                results.append(
                    {
                        "prompt_idx": pidx,
                        "prompt": text,
                        "alpha": float(alpha),
                        "layer": layer,
                        "generation": steered_text.strip(),
                        "mode": "steered",
                    }
                )

    payload = {
        "direction": args.direction,
        "vectors_file": str(args.vectors.resolve()),
        "vectors_results_tag": vectors_results_tag,
        "results_directory": str(out_sweep_json.parent.resolve()),
        "val_pos": str(VAL_POS),
        "val_neg": str(VAL_NEG),
        "prompt_selection": prompt_desc,
        "steering_method": "resteer_v2_val",
        "resteer_batch_size": max(1, RESTEER_BATCH_SIZE),
        "model": MODEL_ID,
        "identify_temperature": IDENTIFY_TEMPERATURE,
        "max_steer_seq_len": MAX_STEER_SEQ_LEN,
        "alphas": alphas,
        "layer_min": layer_lo,
        "layer_max": layer_hi,
        "num_prompts": len(val_rows),
        "results": results,
    }

    out_sweep_json.parent.mkdir(parents=True, exist_ok=True)
    with out_sweep_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved sweep generations to {out_sweep_json}")

    if args.skip_eval:
        return

    run_evaluation(
        out_sweep_json,
        args.direction,
        batch_size=max(1, EVAL_BATCH_SIZE),
        eval_out_json=out_eval_json,
        out_png=out_png,
        out_png_avg=out_png_avg,
    )


if __name__ == "__main__":
    main()
