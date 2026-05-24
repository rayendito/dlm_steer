#!/usr/bin/env python3
"""
Score TIMPA sweep outputs (``extract_vectors/run_timpa_sweep.py``).

Reads ``scores.json`` from a results directory, runs ``utils.eval_utils`` scoring
(same pipeline as ``run_scoring.py``), writes:

- ``scores.json`` updated with per-row scores
- ``scores_scored.csv`` flat table
- ``eval_scores.json`` aggregates + top-5 rankings
- ``heatmaps.png`` per-prompt panels
- ``heatmaps_avg.png`` averaged over prompts

Run from repo root::

    python extract_vectors/score_timpa_sweep.py \\
        --results-dir extract_vectors/results_timpa/results_pos_20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.eval_utils import perplexity, score_labels

EVAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DEFAULT_BATCH_SIZE = 1


def scoring_labels(data: str) -> tuple[str, str]:
    if data == "cats-dogs":
        return "dog", "cat"
    return "positive", "negative"


def target_label(data: str, direction: str) -> str:
    label_1, label_2 = scoring_labels(data)
    if direction == "positive":
        return label_2 if data == "cats-dogs" else label_1
    if direction == "negative":
        return label_1 if data == "cats-dogs" else label_2
    raise ValueError("direction must be positive or negative")


def target_metric_key(data: str, direction: str) -> str:
    return f"prob_{target_label(data, direction)}"


def metric_title(data: str, direction: str) -> str:
    tgt = target_label(data, direction)
    if data == "cats-dogs":
        return f"P({tgt})"
    return f"{tgt.capitalize()} P(token)"


def _build_avg(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]],
    metric_key: str,
) -> dict[tuple[int, float], dict[str, float]]:
    agg: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(
        lambda: {metric_key: [], "perplexity": []}
    )
    for pmap in per_prompt_scores.values():
        for key, s in pmap.items():
            agg[key][metric_key].append(float(s[metric_key]))
            agg[key]["perplexity"].append(float(s["perplexity"]))

    return {
        key: {
            metric_key: sum(vals[metric_key]) / max(1, len(vals[metric_key])),
            "perplexity": sum(vals["perplexity"]) / max(1, len(vals["perplexity"])),
        }
        for key, vals in agg.items()
    }


def _matrices_from_scores(
    scores: dict[tuple[int, float], dict[str, float]],
    metric_key: str,
) -> tuple[list[int], list[float], np.ndarray, np.ndarray, np.ndarray]:
    if not scores:
        return [], [], np.array([]), np.array([]), np.array([])
    layers = sorted({k[0] for k in scores.keys()})
    alphas = sorted({k[1] for k in scores.keys()})
    li_map = {layer: i for i, layer in enumerate(layers)}
    ai_map = {alpha: i for i, alpha in enumerate(alphas)}
    m_tgt = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    m_ppl = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    for (layer, alpha), s in scores.items():
        m_tgt[li_map[layer], ai_map[alpha]] = float(s[metric_key])
        m_ppl[li_map[layer], ai_map[alpha]] = float(s["perplexity"])
    return layers, alphas, m_tgt, m_ppl, m_tgt * m_ppl


def _robust_normalize(m: np.ndarray, lo_pct: float = 0.0, hi_pct: float = 85.0) -> np.ndarray:
    finite = np.isfinite(m)
    if not np.any(finite):
        return np.zeros_like(m)
    vals = m[finite]
    lo = float(np.nanpercentile(vals, lo_pct))
    hi = float(np.nanpercentile(vals, hi_pct))
    if hi < lo:
        lo, hi = hi, lo
    clipped = np.clip(m, lo, hi)
    cmin = float(np.nanmin(clipped[finite]))
    cmax = float(np.nanmax(clipped[finite]))
    if cmax > cmin:
        return (clipped - cmin) / (cmax - cmin)
    return np.zeros_like(m)


def _fmt_alpha(alpha: float) -> str:
    if abs(alpha - round(alpha)) < 1e-9:
        return str(int(round(alpha)))
    return f"{alpha:.3g}"


def _harmonic_norm_target_inv_norm_ppl(
    m_tgt_norm: np.ndarray, m_ppl_norm: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    a = np.asarray(m_tgt_norm, dtype=float)
    b = 1.0 - np.asarray(m_ppl_norm, dtype=float)
    fin = np.isfinite(a) & np.isfinite(b)
    a_c = np.clip(np.where(fin, a, 0.0), 0.0, None)
    b_c = np.clip(np.where(fin, b, 0.0), 0.0, None)
    denom = a_c + b_c
    out = np.full_like(a_c, np.nan, dtype=float)
    good = fin & (denom > eps)
    return np.where(good, (2.0 * a_c * b_c) / denom, np.nan)


def _draw_three(
    axes: list[Any],
    layers: list[int],
    alphas: list[float],
    m_tgt_raw: np.ndarray,
    m_ppl_raw: np.ndarray,
    row_title: str,
    target_title: str,
) -> None:
    m_tgt_plot = _robust_normalize(m_tgt_raw)
    m_ppl_plot = _robust_normalize(m_ppl_raw)
    m_combo = _harmonic_norm_target_inv_norm_ppl(m_tgt_plot, m_ppl_plot)
    fig = axes[0].figure
    for ax, mat, title, cmap, cbar_label in [
        (axes[0], m_tgt_plot, f"{row_title} - {target_title} (normalized)", "magma", "norm prob"),
        (axes[1], m_ppl_plot, f"{row_title} - Perplexity (normalized)", "viridis_r", "norm ppl"),
        (axes[2], m_combo, f"{row_title} - Harmonic mean", "cividis", "harmonic"),
    ]:
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _plot_heatmaps(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]],
    metric_key: str,
    target_title: str,
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
        layers, alphas, m_tgt, m_ppl, _ = _matrices_from_scores(
            per_prompt_scores[pid], metric_key
        )
        _draw_three(list(axes[i]), layers, alphas, m_tgt, m_ppl, f"Prompt {pid}", target_title)
    fig.suptitle(
        "TIMPA sweep: normalized target prob, normalized perplexity, harmonic mean",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {out_png}", flush=True)


def _plot_avg(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    metric_key: str,
    target_title: str,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, alphas, m_tgt, m_ppl, _ = _matrices_from_scores(avg_scores, metric_key)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    _draw_three(list(axes), layers, alphas, m_tgt, m_ppl, "Average (all prompts)", target_title)
    fig.suptitle("TIMPA sweep averaged over val prompts", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved avg heatmap: {out_png}", flush=True)


def _compute_top5(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    metric_key: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "target_metric": metric_key,
        "by_highest_target_prob": [],
        "by_lowest_perplexity": [],
        "total_averaged_cells": len(avg_scores),
    }
    if not avg_scores:
        return base

    cells = list(avg_scores.items())
    by_tgt = sorted(cells, key=lambda kv: float(kv[1][metric_key]), reverse=True)[:5]
    base["by_highest_target_prob"] = [
        {
            "rank": i,
            "layer": layer,
            "alpha": alpha,
            metric_key: float(s[metric_key]),
            "perplexity": float(s["perplexity"]),
        }
        for i, ((layer, alpha), s) in enumerate(by_tgt, start=1)
    ]

    by_ppl = sorted(
        cells,
        key=lambda kv: float(kv[1]["perplexity"])
        if np.isfinite(float(kv[1]["perplexity"]))
        else float("inf"),
    )[:5]
    base["by_lowest_perplexity"] = [
        {
            "rank": i,
            "layer": layer,
            "alpha": alpha,
            metric_key: float(s[metric_key]),
            "perplexity": float(s["perplexity"]),
        }
        for i, ((layer, alpha), s) in enumerate(by_ppl, start=1)
    ]
    return base


def score_sweep(
    results_dir: Path,
    batch_size: int,
    *,
    skip_heatmaps: bool = False,
) -> None:
    scores_path = results_dir / "scores.json"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing {scores_path}")

    with scores_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    data = str(payload.get("data", "imdb"))
    direction = str(payload.get("direction", "negative"))
    rows = [
        r
        for r in payload.get("results", [])
        if isinstance(r, dict) and r.get("mode") == "steered"
    ]
    if not rows:
        raise SystemExit("No steered rows in scores.json.")

    label_1, label_2 = scoring_labels(data)
    tgt_label = target_label(data, direction)
    metric_key = target_metric_key(data, direction)
    tgt_title = metric_title(data, direction)

    eval_model = AutoModelForCausalLM.from_pretrained(EVAL_MODEL_ID).to(DEVICE).eval()
    eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_ID)

    per_sample_scores: dict[tuple[int, int, float], dict[str, float]] = {}
    csv_rows: list[dict[str, Any]] = []

    total = len(rows)
    for start in tqdm(range(0, total, batch_size), desc="eval_utils"):
        batch = rows[start : start + batch_size]
        texts = [str(r.get("generation", "")) for r in batch]
        preds, prob_dict = score_labels(eval_model, eval_tokenizer, texts, label_1, label_2)
        ppls = perplexity(eval_model, eval_tokenizer, texts)

        for i, row in enumerate(batch):
            prompt_idx = int(row.get("prompt_idx", -1))
            layer = int(row["layer"])
            alpha = float(row["alpha"])
            scores = {
                metric_key: float(prob_dict[tgt_label][i]),
                f"prob_{label_1}": float(prob_dict[label_1][i]),
                f"prob_{label_2}": float(prob_dict[label_2][i]),
                "perplexity": float(ppls[i]),
            }
            per_sample_scores[(prompt_idx, layer, alpha)] = scores

            row["predicted_label"] = preds[i]
            row.update(scores)

            csv_rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "layer": layer,
                    "alpha": alpha,
                    "prompt": row.get("prompt", ""),
                    "generation": texts[i],
                    "predicted_label": preds[i],
                    **scores,
                }
            )

    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]] = {}
    for (prompt_idx, layer, alpha), s in per_sample_scores.items():
        per_prompt_scores.setdefault(prompt_idx, {})[(layer, alpha)] = s

    avg_scores = _build_avg(per_prompt_scores, metric_key)
    top5 = _compute_top5(avg_scores, metric_key)

    payload["scorer"] = "utils.eval_utils.score_labels + utils.eval_utils.perplexity"
    payload["eval_model"] = EVAL_MODEL_ID
    payload["score_labels"] = [label_1, label_2]
    payload["target_metric"] = metric_key
    payload["results"] = rows

    with scores_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Updated {scores_path} with per-row scores", flush=True)

    csv_path = results_dir / "scores_scored.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    eval_path = results_dir / "eval_scores.json"
    eval_payload = {
        "scores_file": str(scores_path.resolve()),
        "data": data,
        "direction": direction,
        "scorer": payload["scorer"],
        "eval_model": EVAL_MODEL_ID,
        "score_labels": [label_1, label_2],
        "target_metric": metric_key,
        "results_per_sample": [
            {"prompt_idx": p, "layer": l, "alpha": a, **per_sample_scores[(p, l, a)]}
            for (p, l, a) in sorted(per_sample_scores.keys())
        ],
        "top5_rankings": top5,
    }
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    print(f"Wrote {eval_path}", flush=True)

    if skip_heatmaps:
        return

    try:
        _plot_heatmaps(
            per_prompt_scores,
            metric_key,
            tgt_title,
            results_dir / "heatmaps.png",
        )
        _plot_avg(
            avg_scores,
            metric_key,
            tgt_title,
            results_dir / "heatmaps_avg.png",
        )
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib not installed; skipping heatmaps.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score TIMPA sweep results (eval_utils + heatmaps)."
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing scores.json from run_timpa_sweep.py",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--skip-heatmaps", action="store_true")
    args = ap.parse_args()

    score_sweep(
        args.results_dir.resolve(),
        max(1, args.batch_size),
        skip_heatmaps=args.skip_heatmaps,
    )


if __name__ == "__main__":
    main()
