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

Prompt source (opposite class): steer **toward** the chosen direction using texts from the
other class (IMDB val_pos/val_neg or cats_dogs/val.csv by concept).

``--data`` is inferred from ``--vectors`` (``diffusion-imdb-*`` → imdb,
``diffusion-catdog-*`` → cats-dogs) unless overridden.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Script is run as a file (`python extract_vectors/...`); repo root must be on path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_root = str(_REPO_ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

from transformers import AutoModelForCausalLM, AutoTokenizer

from llada.configuration_llada import LLaDAConfig
from utils.eval_utils import perplexity as eval_perplexity
from utils.eval_utils import score_labels as eval_score_labels
from llada.generate import resteer_v2_val, resteer_v2
from llada.modeling_llada import LLaDAModelLM

# --- hardcoded sweep / IO -------------------------------------------------
IMDB_VAL_POS = _REPO_ROOT / "benchmarks/imdb/val_pos.csv"
IMDB_VAL_NEG = _REPO_ROOT / "benchmarks/imdb/val_neg.csv"
CATS_DOGS_VAL = _REPO_ROOT / "benchmarks/cats_dogs/val.csv"
VAL_PROMPT_LIMIT = 20
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda"
SEED = 42

ALPHAS = [
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    # 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100
    # 2, 3, 4, 5, 6, 7
]

LAYER_MIN = 0
LAYER_MAX = 33
MAX_STEER_SEQ_LEN = 1024
IDENTIFY_TEMPERATURE = 0.0001
RESTEER_STEPS = 1
REFILL_STEPS = 1
PERPLEXITY_THRESHOLD = 10000
SENTIMENT_THRESHOLD = 0.1

# Writes under ``extract_vectors/results_{pos|neg}_{tag}/`` where ``tag`` is from ``--vectors``
# and ``pos``/``neg`` from ``--direction`` (e.g. ``...-n20.pt`` + negative → ``results_neg_20``).
RESULTS_PARENT = Path("extract_vectors") / "results"
EVAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EVAL_BATCH_SIZE = 1
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


@dataclass(frozen=True)
class SweepConfig:
    data: str
    direction: str
    results_abbrev: str
    label_positive: str
    label_negative: str
    metric_key: str
    steer_flip: bool


def infer_data_from_vectors(vectors_path: Path, data_cli: str | None) -> str:
    if data_cli is not None:
        if data_cli not in ("imdb", "cats-dogs"):
            raise ValueError(f"--data must be imdb or cats-dogs, got {data_cli!r}")
        return data_cli
    stem = vectors_path.stem.lower()
    if "catdog" in stem:
        return "cats-dogs"
    return "imdb"


def resolve_sweep_config(vectors_path: Path, direction: str, data: str) -> SweepConfig:
    d = direction.strip().lower()
    if data == "cats-dogs":
        label_pos, label_neg = "cat", "dog"
        key_pos, key_neg = "cat_sentiment_prob", "dog_sentiment_prob"
        if d in ("positive", "cats", "cat"):
            return SweepConfig(
                data=data,
                direction="cats",
                results_abbrev="cats",
                label_positive=label_pos,
                label_negative=label_neg,
                metric_key=key_pos,
                steer_flip=True,
            )
        if d in ("negative", "dogs", "dog"):
            return SweepConfig(
                data=data,
                direction="dogs",
                results_abbrev="dogs",
                label_positive=label_pos,
                label_negative=label_neg,
                metric_key=key_neg,
                steer_flip=False,
            )
        raise ValueError(
            f"For cats-dogs data, --direction must be cats/dogs (or positive/negative aliases), "
            f"got {direction!r}"
        )

    if data == "imdb":
        label_pos, label_neg = "positive", "negative"
        key_pos, key_neg = "positive_sentiment_prob", "negative_sentiment_prob"
        if d in ("cats", "cat", "dogs", "dog"):
            raise ValueError(
                f"For imdb data, use --direction positive or negative, got {direction!r}"
            )
        if d == "positive":
            return SweepConfig(
                data=data,
                direction="positive",
                results_abbrev="pos",
                label_positive=label_pos,
                label_negative=label_neg,
                metric_key=key_pos,
                steer_flip=True,
            )
        if d == "negative":
            return SweepConfig(
                data=data,
                direction="negative",
                results_abbrev="neg",
                label_positive=label_pos,
                label_negative=label_neg,
                metric_key=key_neg,
                steer_flip=False,
            )
        raise ValueError(
            f"For imdb data, --direction must be positive or negative, got {direction!r}"
        )

    raise ValueError(f"Unknown data: {data!r}")


def _load_eval_classifier(device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(EVAL_MODEL_ID).eval().to(device)
    return model, tokenizer


def resolve_result_paths(
    vectors_path: Path, cfg: SweepConfig
) -> tuple[Path, Path, Path, Path, str]:
    tag = results_tag_from_vectors_path(vectors_path)
    out_dir = RESULTS_PARENT / f"results_{cfg.results_abbrev}_{tag}"
    return (
        out_dir / "scores.json",
        out_dir / "eval_scores.json",
        out_dir / "heatmaps.png",
        out_dir / "heatmaps_avg.png",
        tag,
    )


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def _load_imdb_texts(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"{path} must have a 'text' column")
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        raise ValueError(f"No texts loaded from {path}")
    return texts


def _load_cats_dogs_texts(path: Path, concept: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    want = concept.strip().lower()
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "text" not in reader.fieldnames:
            raise ValueError(f"{path} must have 'text' and 'concept' columns")
        for row in reader:
            if (row.get("concept") or "").strip().lower() != want:
                continue
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    if not texts:
        raise ValueError(f"No {concept} texts loaded from {path}")
    return texts


def load_steering_prompt_texts(cfg: SweepConfig) -> tuple[list[tuple[int, str, str]], str]:
    """
    Load val texts from the class *opposite* the steering target.

    IMDB: positive → val_neg; negative → val_pos.
    cats-dogs: cats → dog rows; dogs → cat rows.
    """
    if cfg.data == "imdb":
        if cfg.direction == "positive":
            source_label, texts = "neg", _load_imdb_texts(IMDB_VAL_NEG)
            desc = "imdb val_neg (steer toward positive)"
        else:
            source_label, texts = "pos", _load_imdb_texts(IMDB_VAL_POS)
            desc = "imdb val_pos (steer toward negative)"
    elif cfg.data == "cats-dogs":
        if cfg.direction == "cats":
            source_label, texts = "dog", _load_cats_dogs_texts(CATS_DOGS_VAL, "dog")
            desc = "cats_dogs val dog rows (steer toward cat)"
        else:
            source_label, texts = "cat", _load_cats_dogs_texts(CATS_DOGS_VAL, "cat")
            desc = "cats_dogs val cat rows (steer toward dog)"
    else:
        raise ValueError(f"Unknown data: {cfg.data!r}")

    limited = texts[:VAL_PROMPT_LIMIT]
    rows = [(i, t, source_label) for i, t in enumerate(limited)]
    return rows, desc


def build_steer_bases(
    vectors_path: Path,
    cfg: SweepConfig,
    device: str,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], int]:
    if not vectors_path.is_file():
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")
    sentiment_vectors = torch.load(vectors_path, map_location=device)
    pos_vectors = sentiment_vectors["positive"]
    neg_vectors = sentiment_vectors["negative"]
    steer_all = tuple(
        neg_vectors[i].to(dtype=dtype, device=device)
        - pos_vectors[i].to(dtype=dtype, device=device)
        for i in range(len(pos_vectors))
    )
    if cfg.steer_flip:
        steer_all = tuple(-v for v in steer_all)
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
    cfg: SweepConfig,
    *,
    eval_model: Any,
    eval_tokenizer: Any,
    batch_size: int,
    eval_out_json: Path,
    out_png: Path,
    out_png_avg: Path,
) -> None:
    tgt_label = (
        cfg.label_negative
        if cfg.direction in ("negative", "dogs")
        else cfg.label_positive
    )
    sentiment_key = cfg.metric_key
    sentiment_title = f"{tgt_label} P(token)"

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
        desc="eval_utils batches",
        total=n_batches,
        file=sys.stderr,
    ):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        texts = [str(r.get("generation", "")) for r in batch]
        _, prob_dict = eval_score_labels(
            eval_model,
            eval_tokenizer,
            texts,
            cfg.label_positive,
            cfg.label_negative,
        )
        ppls = eval_perplexity(eval_model, eval_tokenizer, texts)
        tgt_probs = prob_dict[tgt_label]

        for i, row in enumerate(batch):
            prompt_idx = int(row.get("prompt_idx", -1))
            layer = int(row["layer"])
            alpha = float(row["alpha"])
            s = {
                sentiment_key: float(tgt_probs[i]),
                "perplexity": float(ppls[i]),
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

    avg_scores = _build_avg(per_prompt_scores, sentiment_key)
    top5_rankings = _compute_top5_rankings(avg_scores, sentiment_key)

    out_payload = {
        "sweep_file": str(sweep_file),
        "data": cfg.data,
        "direction": cfg.direction,
        "scorer": (
            f"utils.eval_utils.score_labels ({cfg.label_positive}/{cfg.label_negative}) "
            "+ perplexity"
        ),
        "eval_model": EVAL_MODEL_ID,
        "eval_labels": [cfg.label_positive, cfg.label_negative],
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
        "top5_rankings": top5_rankings,
    }
    eval_out_json.parent.mkdir(parents=True, exist_ok=True)
    with eval_out_json.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Wrote {eval_out_json}", file=sys.stderr)

    _plot_heatmaps(per_prompt_scores, sentiment_key, sentiment_title, out_png)
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


def _harmonic_sent_inv_norm_ppl(M_sent: Any, M_ppl_norm_plot: Any, eps: float = 1e-8) -> Any:
    """
    Element-wise harmonic mean of sentiment (higher better) and (1 − normalized_ppl)
    (higher perplexity quality after flip). NaN where inputs are non-finite.
    """
    a = np.asarray(M_sent, dtype=float)
    b = 1.0 - np.asarray(M_ppl_norm_plot, dtype=float)
    fin = np.isfinite(a) & np.isfinite(b)
    a_c = np.clip(np.where(fin, a, 0.0), 0.0, None)
    b_c = np.clip(np.where(fin, b, 0.0), 0.0, None)
    denom = a_c + b_c
    out = np.full_like(a_c, np.nan, dtype=float)
    good = fin & (denom > eps)
    out = np.where(good, (2.0 * a_c * b_c) / denom, np.nan)
    return out


def _harmonic_sent_inv_norm_ppl_scalar(sent: float, ppl_norm: float, eps: float = 1e-8) -> float:
    a = max(float(sent), 0.0)
    b = max(1.0 - float(ppl_norm), 0.0)
    d = a + b
    if d <= eps:
        return 0.0
    return (2.0 * a * b) / d


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
    M_combo = _harmonic_sent_inv_norm_ppl(M_sent, M_ppl_plot)
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
            f"{row_title} - Harmonic mean (sentiment, inv norm ppl)",
            "cividis",
            0.0,
            1.0,
            "harmonic mean",
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
        "Val sweep: sentiment, perplexity, harmonic mean (sent × inv norm ppl)",
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
    fig.suptitle("Val sweep averaged: sentiment, perplexity, harmonic mean", fontsize=12)
    out_png_avg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_avg, dpi=150)
    plt.close(fig)
    print(f"Saved avg heatmap: {out_png_avg}", file=sys.stderr)


def _meets_score_thresholds(s: dict[str, float], sentiment_key: str) -> bool:
    """Keep rows with perplexity at or below cap and sentiment at or above floor."""
    ppl = float(s["perplexity"])
    sent = float(s[sentiment_key])
    if not np.isfinite(ppl) or not np.isfinite(sent):
        return False
    if ppl > PERPLEXITY_THRESHOLD:
        return False
    if sent < SENTIMENT_THRESHOLD:
        return False
    return True


def _compute_top5_rankings(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    sentiment_key: str,
) -> dict[str, Any]:
    """Top-5 lists after threshold filter; JSON-serializable."""
    base: dict[str, Any] = {
        "perplexity_threshold": PERPLEXITY_THRESHOLD,
        "sentiment_threshold": SENTIMENT_THRESHOLD,
        "sentiment_metric": sentiment_key,
        "filter": f"ppl <= {PERPLEXITY_THRESHOLD:g}, {sentiment_key} >= {SENTIMENT_THRESHOLD:g}",
        "by_highest_sentiment": [],
        "by_lowest_perplexity": [],
        "by_harmonic_mean_unique_layer": [],
        "combined_score": "harmonic_mean(sentiment, 1 - ppl_norm_on_filtered)",
        "total_averaged_cells": len(avg_scores),
        "cells_passing_filter": 0,
        "no_rows_passed_thresholds": True,
    }
    if not avg_scores:
        base["skipped_reason"] = "no_avg_scores"
        return base

    cells = list(avg_scores.items())
    filtered = [kv for kv in cells if _meets_score_thresholds(kv[1], sentiment_key)]
    base["cells_passing_filter"] = len(filtered)

    if not filtered:
        base["skipped_reason"] = "no_rows_pass_thresholds"
        return base

    base["no_rows_passed_thresholds"] = False

    def _row(
        rank: int,
        layer: int,
        alpha: float,
        s: dict[str, float],
        harmonic_mean: float | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rank": rank,
            "layer": layer,
            "alpha": alpha,
            sentiment_key: float(s[sentiment_key]),
            "perplexity": float(s["perplexity"]),
        }
        if harmonic_mean is not None:
            out["harmonic_mean"] = float(harmonic_mean)
        return out

    by_sent = sorted(
        filtered,
        key=lambda kv: float(kv[1][sentiment_key]),
        reverse=True,
    )[:5]
    base["by_highest_sentiment"] = [
        _row(i, layer, alpha, s, None)
        for i, ((layer, alpha), s) in enumerate(by_sent, start=1)
    ]

    def _ppl_sort_key(kv: tuple[tuple[int, float], dict[str, float]]) -> float:
        p = float(kv[1]["perplexity"])
        return p if np.isfinite(p) else float("inf")

    by_ppl = sorted(filtered, key=_ppl_sort_key)[:5]
    base["by_lowest_perplexity"] = [
        _row(i, layer, alpha, s, None)
        for i, ((layer, alpha), s) in enumerate(by_ppl, start=1)
    ]

    ppl_vals = np.array([float(v["perplexity"]) for _, v in filtered], dtype=float)
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

    ranked: list[tuple[float, int, float, dict[str, float]]] = []
    for i, ((layer, alpha), s) in enumerate(filtered):
        sent = float(s[sentiment_key])
        hm = _harmonic_sent_inv_norm_ppl_scalar(sent, float(ppl_norm[i]))
        ranked.append((hm, layer, alpha, s))
    ranked.sort(key=lambda x: x[0], reverse=True)

    topk: list[tuple[float, int, float, dict[str, float]]] = []
    used_layers: set[int] = set()
    for hm, layer, alpha, s in ranked:
        if layer in used_layers:
            continue
        topk.append((hm, layer, alpha, s))
        used_layers.add(layer)
        if len(topk) == 5:
            break

    base["by_harmonic_mean_unique_layer"] = [
        _row(i, layer, alpha, s, hm)
        for i, (hm, layer, alpha, s) in enumerate(topk, start=1)
    ]
    return base


def _print_top5(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    sentiment_key: str,
) -> None:
    if not avg_scores:
        print("No averaged scores for top-5.", file=sys.stderr)
        return
    d = _compute_top5_rankings(avg_scores, sentiment_key)
    thr_note = (
        f"ppl ≤ {PERPLEXITY_THRESHOLD:g}, {sentiment_key} ≥ {SENTIMENT_THRESHOLD:g}"
    )

    if d["no_rows_passed_thresholds"]:
        print(
            f"\nNo averaged combinations pass thresholds ({thr_note}). Skipping top-5 lists.",
            file=sys.stderr,
        )
        return

    print(
        f"\nTop 5 by highest {sentiment_key} (combination-wise; {thr_note}):",
        file=sys.stderr,
    )
    for row in d["by_highest_sentiment"]:
        print(
            f"{row['rank']}. layer={row['layer']}, alpha={row['alpha']:g}, "
            f"{sentiment_key}={row[sentiment_key]:.4f}, ppl={row['perplexity']:.4f}",
            file=sys.stderr,
        )

    print(
        f"\nTop 5 by lowest perplexity (combination-wise; {thr_note}):",
        file=sys.stderr,
    )
    for row in d["by_lowest_perplexity"]:
        print(
            f"{row['rank']}. layer={row['layer']}, alpha={row['alpha']:g}, "
            f"ppl={row['perplexity']:.4f}, {sentiment_key}={row[sentiment_key]:.4f}",
            file=sys.stderr,
        )

    print(
        "\nTop 5 by harmonic mean (unique layers, higher is better; "
        f"{thr_note}):",
        file=sys.stderr,
    )
    for row in d["by_harmonic_mean_unique_layer"]:
        print(
            f"{row['rank']}. layer={row['layer']}, alpha={row['alpha']:g}, "
            f"harmonic_mean={row['harmonic_mean']:.4f}, {sentiment_key}={row[sentiment_key]:.4f}, "
            f"ppl={row['perplexity']:.4f}",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Val steering sweep (LLaDA resteer_v2_val on full val texts). "
        "Edit module-level constants for paths, alphas, layers, resteer hyperparameters, outputs."
    )
    ap.add_argument(
        "--direction",
        choices=["positive", "negative", "cats", "dogs"],
        required=True,
        help=(
            "Steering target. IMDB: positive/negative. cats-dogs: cats/dogs "
            "(positive→cats, negative→dogs)."
        ),
    )
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument(
        "--data",
        choices=["imdb", "cats-dogs"],
        default=None,
        help="Dataset (default: infer from --vectors filename).",
    )
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    vectors_path = Path(args.vectors)
    data = infer_data_from_vectors(vectors_path, args.data)
    cfg = resolve_sweep_config(vectors_path, args.direction, data)

    (
        out_sweep_json,
        out_eval_json,
        out_png,
        out_png_avg,
        vectors_results_tag,
    ) = resolve_result_paths(vectors_path, cfg)
    print(
        f"Outputs → {out_sweep_json.parent} "
        f"(data={cfg.data}, direction={cfg.direction}, vectors tag={vectors_results_tag})",
        flush=True,
    )

    device = DEVICE
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    val_rows, prompt_desc = load_steering_prompt_texts(cfg)
    texts = [t for _, t, _ in val_rows]
    prompt_idxs = [i for i, _, _ in val_rows]
    print(f"Sweep prompts: {len(val_rows)} rows — {prompt_desc}", flush=True)

    alphas = list(ALPHAS)
    steer_bases, num_layers = build_steer_bases(
        vectors_path, cfg, device, torch.bfloat16
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
                resteer_v2(
                    model,
                    tokenized_inputs,
                    steer_vectors,
                    resteer_steps=RESTEER_STEPS,
                    refill_steps=REFILL_STEPS,
                )
                steered_ids = tokenized_inputs["input_ids"]
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
        "data": cfg.data,
        "direction": cfg.direction,
        "vectors_file": str(vectors_path.resolve()),
        "vectors_results_tag": vectors_results_tag,
        "results_folder": out_sweep_json.parent.name,
        "results_directory": str(out_sweep_json.parent.resolve()),
        "val_pos": str(IMDB_VAL_POS),
        "val_neg": str(IMDB_VAL_NEG),
        "cats_dogs_val": str(CATS_DOGS_VAL),
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

    print(
        f"Eval classifier: {EVAL_MODEL_ID}  labels={cfg.label_positive}/{cfg.label_negative}  "
        f"metric={cfg.metric_key}",
        flush=True,
    )
    eval_model, eval_tokenizer = _load_eval_classifier(device)

    run_evaluation(
        out_sweep_json,
        cfg,
        eval_model=eval_model,
        eval_tokenizer=eval_tokenizer,
        batch_size=max(1, EVAL_BATCH_SIZE),
        eval_out_json=out_eval_json,
        out_png=out_png,
        out_png_avg=out_png_avg,
    )


if __name__ == "__main__":
    main()
