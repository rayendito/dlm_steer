#!/usr/bin/env python3
"""
TIMPA-style staged hyperparameter search built on top of ``sandbox.py`` / ``resteer_v2``.

Why this script exists
----------------------
The old ``eval_timpa.py`` only dumped text evolutions. That was useful for inspection, but
it did not answer the more important question:

1. for a given ``refill_steps`` value, which outer steering step is best within a full
   fixed-length resteer trajectory?
2. after picking the best ``refill_steps``, which temperature setting is best?
3. after picking that best steering setup, which sentence-length threshold is best, and
   how do the final sentence-length bins behave?

This script does that in two stages so the search stays manageable.
The parameter search schedule is intentionally hard-coded so the experiment can
automatically choose the next stage based on the best result from the previous stage.

Stage 1
  Fix ``resteer_steps`` to a large budget (default: 32).
  Sweep only ``refill_steps``.
  For each refill setting, run the full resteer trajectory and evaluate *every* outer
  steering step with ``eval_dito.score_labels`` and ``eval_dito.perplexity``.
  Then pick the best step inside that trajectory.

Stage 2
  Freeze the best ``refill_steps`` from stage 1.
  Sweep temperature values and again score every outer steering step.
  Pick the best temperature from that search.

Stage 3
  Freeze the best steering setup from stage 2.
  Search sentence-length thresholds as a scoring parameter.
  Then report final sentence-length bins using that same best steering setup.

Notes on evaluation
-------------------
``eval_dito.score_labels`` returns the classifier probability for the sentiment token.
For ``--steer-direction negative`` we maximize ``negative`` probability.
For ``--steer-direction positive`` we maximize ``positive`` probability.

Perplexity is lower-is-better, so we robust-normalize it and combine it with sentiment
using a harmonic mean:

  harmonic_mean(sentiment_prob, 1 - normalized_perplexity)

That matches the repo's existing sweep ranking style.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from eval_dito import perplexity as dito_perplexity
from eval_dito import score_labels
from llada.configuration_llada import LLaDAConfig
from llada.generate import resteer_v2
from llada.modeling_llada import LLaDAModelLM


SEED = 42
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_RESTEER_STEPS = 32
DEFAULT_REFILL_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
DEFAULT_SAMPLING_TEMP_GRID = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0,
]
DEFAULT_SENTENCE_LENGTH_GRID = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
DEFAULT_SENTENCE_LENGTH_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 10**9]


@dataclass(frozen=True)
class PromptRow:
    prompt_idx: int
    text: str
    word_count: int
    token_count: int


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_layer_list(raw: str) -> list[int]:
    return sorted({int(x.strip()) for x in raw.split(",") if x.strip()})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def load_prompts(dataset_path: Path, tokenizer: AutoTokenizer, max_seq_len: int) -> list[PromptRow]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts: list[str] = []
    if dataset_path.suffix.lower() == ".csv":
        with dataset_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise ValueError(f"{dataset_path} must contain a 'text' column")
            for row in reader:
                text = (row.get("text") or "").strip()
                if text:
                    texts.append(text)
    else:
        with dataset_path.open(encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(text)

    rows: list[PromptRow] = []
    for idx, text in enumerate(texts):
        token_count = len(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
            )["input_ids"]
        )
        rows.append(
            PromptRow(
                prompt_idx=idx,
                text=text,
                word_count=len(text.split()),
                token_count=token_count,
            )
        )
    if not rows:
        raise ValueError(f"No prompts loaded from {dataset_path}")
    return rows


def build_steer_vectors(
    steer_vectors_path: Path,
    direction: str,
    layers: list[int],
    alpha: float,
    device: str,
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    if not steer_vectors_path.is_file():
        raise FileNotFoundError(f"Steer vectors not found: {steer_vectors_path}")

    sentiment_vectors = torch.load(steer_vectors_path, map_location=device)
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
        raise ValueError("--steer-direction must be positive or negative")

    invalid_layers = [layer for layer in layers if layer < 0 or layer >= len(steer_all)]
    if invalid_layers:
        raise ValueError(
            f"Invalid steer layers {invalid_layers}; available range is 0..{len(steer_all) - 1}"
        )
    return {layer: float(alpha) * steer_all[layer] for layer in layers}


def load_model_and_tokenizer(device: str) -> tuple[LLaDAModelLM, AutoTokenizer]:
    config = LLaDAConfig.from_pretrained(MODEL_ID)
    model = LLaDAModelLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return model, tokenizer


def robust_normalize(values: list[float], lo_pct: float = 0.0, hi_pct: float = 85.0) -> list[float]:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return [0.0 for _ in values]

    vals = arr[finite]
    lo = float(np.nanpercentile(vals, lo_pct))
    hi = float(np.nanpercentile(vals, hi_pct))
    if hi < lo:
        lo, hi = hi, lo
    clipped = np.clip(arr, lo, hi)
    cmin = float(np.nanmin(clipped[finite]))
    cmax = float(np.nanmax(clipped[finite]))
    if cmax <= cmin:
        return [0.0 for _ in values]
    return [float(x) for x in ((clipped - cmin) / (cmax - cmin))]


def harmonic_sent_inv_ppl_scalar(sent: float, ppl_norm: float, eps: float = 1e-8) -> float:
    a = max(float(sent), 0.0)
    b = max(1.0 - float(ppl_norm), 0.0)
    denom = a + b
    if denom <= eps:
        return 0.0
    return (2.0 * a * b) / denom


def score_texts(
    texts: list[str],
    direction: str,
) -> list[dict[str, float]]:
    _, p_pos, p_neg = score_labels(texts)
    ppls = dito_perplexity(texts)
    out: list[dict[str, float]] = []
    for i in range(len(texts)):
        pos = float(p_pos[i].item())
        neg = float(p_neg[i].item())
        ppl = float(ppls[i].item())
        target = neg if direction == "negative" else pos
        out.append(
            {
                "positive_sentiment": pos,
                "negative_sentiment": neg,
                "target_sentiment": target,
                "perplexity": ppl,
            }
        )
    return out


def summarize_scored_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "avg_positive_sentiment": 0.0,
            "avg_negative_sentiment": 0.0,
            "avg_target_sentiment": 0.0,
            "avg_perplexity": 0.0,
            "perplexity_norm_avg": 0.0,
            "harmonic_mean_avg": 0.0,
            "combined_score": 0.0,
            "selection_metric": "harmonic_mean(target_sentiment, 1 - normalized_perplexity)",
        }

    ppl_norm = robust_normalize([float(r["perplexity"]) for r in rows])
    combined = [
        harmonic_sent_inv_ppl_scalar(float(r["target_sentiment"]), ppl_norm[i])
        for i, r in enumerate(rows)
    ]
    return {
        "count": len(rows),
        "avg_positive_sentiment": float(np.mean([r["positive_sentiment"] for r in rows])),
        "avg_negative_sentiment": float(np.mean([r["negative_sentiment"] for r in rows])),
        "avg_target_sentiment": float(np.mean([r["target_sentiment"] for r in rows])),
        "avg_perplexity": float(np.mean([r["perplexity"] for r in rows])),
        "perplexity_norm_avg": float(np.mean(ppl_norm)),
        "harmonic_mean_avg": float(np.mean(combined)),
        "combined_score": float(np.mean(combined)),
        "selection_metric": "harmonic_mean(target_sentiment, 1 - normalized_perplexity)",
    }


def run_resteer_pair(
    *,
    model: LLaDAModelLM,
    tokenizer: AutoTokenizer,
    prompts: list[PromptRow],
    steer_vectors: dict[int, torch.Tensor],
    direction: str,
    resteer_steps: int,
    refill_steps: int,
    batch_size: int,
    max_seq_len: int,
    sampling_temp: float,
    identify_temp: float,
    alpha_decay: bool,
    device: str,
) -> dict[str, Any]:
    per_step_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    total = len(prompts)

    for start in tqdm(
        range(0, total, batch_size),
        desc=f"resteer={resteer_steps} refill={refill_steps}",
        leave=False,
    ):
        chunk = prompts[start : start + batch_size]
        texts = [row.text for row in chunk]
        tokenized_inputs = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            step_results = resteer_v2(
                model,
                tokenized_inputs,
                steer_vectors,
                resteer_steps=resteer_steps,
                refill_steps=refill_steps,
                sampling_temp=sampling_temp,
                identify_temp=identify_temp,
                alpha_decay=alpha_decay,
            )

        for step_result in step_results:
            step_idx = int(step_result["resteer_step"])
            decoded = tokenizer.batch_decode(
                step_result["after"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            scored = score_texts(decoded, direction)
            for row, generation, metrics in zip(chunk, decoded, scored, strict=True):
                per_step_rows[step_idx].append(
                    {
                        "prompt_idx": row.prompt_idx,
                        "prompt": row.text,
                        "word_count": row.word_count,
                        "token_count": row.token_count,
                        "resteer_steps": resteer_steps,
                        "refill_steps": refill_steps,
                        "actual_step": step_idx + 1,
                        "generation": generation.strip(),
                        **metrics,
                    }
                )

    step_summaries: list[dict[str, Any]] = []
    for step_idx in sorted(per_step_rows.keys()):
        summary = summarize_scored_rows(per_step_rows[step_idx])
        step_summaries.append(
            {
                "actual_step": step_idx + 1,
                **summary,
            }
        )

    if not step_summaries:
        raise RuntimeError("No step summaries produced")

    best_step = max(step_summaries, key=lambda x: float(x["combined_score"]))
    return {
        "resteer_steps": resteer_steps,
        "refill_steps": refill_steps,
        "step_summaries": step_summaries,
        "best_step_summary": best_step,
        "rows_by_step": {str(k + 1): v for k, v in sorted(per_step_rows.items())},
    }


def build_stage1_matrix(pair_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for item in pair_results:
        best = item["best_step_summary"]
        matrix.append(
            {
                "resteer_steps": item["resteer_steps"],
                "refill_steps": item["refill_steps"],
                "best_actual_step": best["actual_step"],
                "selection_metric": best["selection_metric"],
                "harmonic_mean_avg": best["harmonic_mean_avg"],
                "combined_score": best["combined_score"],
                "avg_target_sentiment": best["avg_target_sentiment"],
                "avg_perplexity": best["avg_perplexity"],
            }
        )
    matrix.sort(key=lambda x: x["refill_steps"])
    return matrix


def build_stage2_matrix(temp_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for item in temp_results:
        best = item["best_step_summary"]
        matrix.append(
            {
                "sampling_temp": item["sampling_temp"],
                "refill_steps": item["refill_steps"],
                "best_actual_step": best["actual_step"],
                "selection_metric": best["selection_metric"],
                "harmonic_mean_avg": best["harmonic_mean_avg"],
                "combined_score": best["combined_score"],
                "avg_target_sentiment": best["avg_target_sentiment"],
                "avg_perplexity": best["avg_perplexity"],
            }
        )
    matrix.sort(key=lambda x: x["sampling_temp"])
    return matrix


def plot_search_heatmap(
    *,
    x_labels: list[str],
    y_labels: list[str],
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = max(8, 0.6 * max(1, len(x_labels)))
    fig_h = max(5, 0.4 * max(1, len(y_labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", origin="lower", interpolation="nearest", cmap="cividis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            ax.text(xi, yi, f"{values[yi, xi]:.3f}", ha="center", va="center", fontsize=7, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="harmonic mean")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, format="jpg")
    plt.close(fig)


def save_stage_heatmaps(
    *,
    exp_name: str,
    stage1_results: list[dict[str, Any]],
    stage2_results: list[dict[str, Any]],
    stage3_search: dict[str, Any],
) -> list[str]:
    image_dir = Path("results") / "image" / exp_name
    written: list[str] = []

    if stage1_results:
        refill_labels = [str(int(r["refill_steps"])) for r in stage1_results]
        max_step = max(len(r["step_summaries"]) for r in stage1_results)
        stage1_mat = np.full((max_step, len(stage1_results)), np.nan, dtype=float)
        for xi, result in enumerate(stage1_results):
            by_step = {int(s["actual_step"]): float(s["harmonic_mean_avg"]) for s in result["step_summaries"]}
            for step in range(1, max_step + 1):
                stage1_mat[step - 1, xi] = by_step.get(step, np.nan)
        out = image_dir / "stage1_refill_vs_step_hm.jpg"
        plot_search_heatmap(
            x_labels=refill_labels,
            y_labels=[str(i) for i in range(1, max_step + 1)],
            values=stage1_mat,
            title="Stage 1: refill_steps vs resteer step",
            xlabel="refill_steps",
            ylabel="outer resteer step",
            out_path=out,
        )
        written.append(str(out))

    if stage2_results:
        temp_labels = [f"{float(r['sampling_temp']):g}" for r in stage2_results]
        max_step = max(len(r["step_summaries"]) for r in stage2_results)
        stage2_mat = np.full((max_step, len(stage2_results)), np.nan, dtype=float)
        for xi, result in enumerate(stage2_results):
            by_step = {int(s["actual_step"]): float(s["harmonic_mean_avg"]) for s in result["step_summaries"]}
            for step in range(1, max_step + 1):
                stage2_mat[step - 1, xi] = by_step.get(step, np.nan)
        out = image_dir / "stage2_temp_vs_step_hm.jpg"
        plot_search_heatmap(
            x_labels=temp_labels,
            y_labels=[str(i) for i in range(1, max_step + 1)],
            values=stage2_mat,
            title="Stage 2: sampling_temp vs resteer step",
            xlabel="sampling_temp",
            ylabel="outer resteer step",
            out_path=out,
        )
        written.append(str(out))

    threshold_results = list(stage3_search.get("threshold_results", []))
    if threshold_results:
        vals = np.array([[float(r["weighted_combined_score"]) for r in threshold_results]], dtype=float)
        out = image_dir / "stage3_sentence_length_threshold_hm.jpg"
        plot_search_heatmap(
            x_labels=[str(int(r["sentence_length_threshold"])) for r in threshold_results],
            y_labels=["weighted score"],
            values=vals,
            title="Stage 3: sentence-length threshold search",
            xlabel="sentence length threshold (words)",
            ylabel="metric",
            out_path=out,
        )
        written.append(str(out))

    return written


def subset_rows_for_best_step(
    pair_result: dict[str, Any],
    actual_step: int,
) -> list[dict[str, Any]]:
    return list(pair_result["rows_by_step"][str(actual_step)])


def evaluate_sentence_length_thresholds(
    best_pair_result: dict[str, Any],
    threshold_grid: list[int],
) -> dict[str, Any]:
    best_step = int(best_pair_result["best_step_summary"]["actual_step"])
    rows = subset_rows_for_best_step(best_pair_result, best_step)

    threshold_results: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        short_rows = [r for r in rows if int(r["word_count"]) <= threshold]
        long_rows = [r for r in rows if int(r["word_count"]) > threshold]
        short_summary = summarize_scored_rows(short_rows)
        long_summary = summarize_scored_rows(long_rows)
        total = max(1, len(rows))
        weighted = (
            short_summary["combined_score"] * len(short_rows)
            + long_summary["combined_score"] * len(long_rows)
        ) / total
        threshold_results.append(
            {
                "sentence_length_threshold": threshold,
                "weighted_combined_score": float(weighted),
                "short_bin": {
                    "name": f"<= {threshold} words",
                    **short_summary,
                },
                "long_bin": {
                    "name": f"> {threshold} words",
                    **long_summary,
                },
            }
        )

    best_threshold = max(threshold_results, key=lambda x: float(x["weighted_combined_score"]))
    return {
        "best_actual_step": best_step,
        "threshold_results": threshold_results,
        "best_threshold_result": best_threshold,
    }


def evaluate_sentence_length_bins(
    best_pair_result: dict[str, Any],
    bins: list[int],
) -> dict[str, Any]:
    if sorted(bins) != bins:
        raise ValueError("--sentence-length-bins must be sorted ascending")
    if len(bins) < 2:
        raise ValueError("--sentence-length-bins needs at least two edges")

    best_step = int(best_pair_result["best_step_summary"]["actual_step"])
    rows = subset_rows_for_best_step(best_pair_result, best_step)

    out_bins: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        bucket = [r for r in rows if lo <= int(r["word_count"]) < hi]
        out_bins.append(
            {
                "word_count_range": [lo, hi],
                "label": f"[{lo}, {hi}) words",
                **summarize_scored_rows(bucket),
            }
        )
    return {
        "best_actual_step": best_step,
        "bins": out_bins,
    }


def print_stage1_summary(best_pair_result: dict[str, Any]) -> None:
    best = best_pair_result["best_step_summary"]
    print(
        (
            "Stage 1 best -> "
            f"resteer_steps={best_pair_result['resteer_steps']}, "
            f"refill_steps={best_pair_result['refill_steps']}, "
            f"actual_step={best['actual_step']}, "
            f"hm={best['harmonic_mean_avg']:.4f}, "
            f"target_sent={best['avg_target_sentiment']:.4f}, "
            f"ppl={best['avg_perplexity']:.4f}"
        )
    )


def print_stage2_summary(
    best_temp_result: dict[str, Any],
    sentence_length_search: dict[str, Any],
    sentence_length_bin_eval: dict[str, Any],
) -> None:
    best_step = best_temp_result["best_step_summary"]
    print(
        (
            "Stage 2 best temp -> "
            f"sampling_temp={best_temp_result['sampling_temp']}, "
            f"refill_steps={best_temp_result['refill_steps']}, "
            f"actual_step={best_step['actual_step']}, "
            f"hm={best_step['harmonic_mean_avg']:.4f}, "
            f"target_sent={best_step['avg_target_sentiment']:.4f}, "
            f"ppl={best_step['avg_perplexity']:.4f}"
        )
    )
    best_thr = sentence_length_search["best_threshold_result"]
    print(
        (
            "Stage 3 best sentence-length threshold -> "
            f"sentence_length<={best_thr['sentence_length_threshold']} split, "
            f"weighted_combined={best_thr['weighted_combined_score']:.4f}"
        )
    )
    for bucket in sentence_length_bin_eval["bins"]:
        print(
            (
                f"  {bucket['label']}: "
                f"n={bucket['count']}, "
                f"hm={bucket['harmonic_mean_avg']:.4f}, "
                f"target_sent={bucket['avg_target_sentiment']:.4f}, "
                f"ppl={bucket['avg_perplexity']:.4f}"
            )
        )


def get_hardcoded_search_plan() -> dict[str, Any]:
    """
    Hard-coded staged search plan.

    Stage 1:
      Keep ``resteer_steps`` fixed and search only ``refill_steps``.

    Stage 2:
      Reuse the best stage-1 steering setup and search temperature.

    Stage 3:
      Reuse the best stage-2 steering setup and search sentence length as a scoring
      parameter using the pre-declared threshold sweep, then report final bins.
    """
    return {
        "resteer_steps": DEFAULT_RESTEER_STEPS,
        "refill_steps_grid": list(DEFAULT_REFILL_GRID),
        "sampling_temp_grid": list(DEFAULT_SAMPLING_TEMP_GRID),
        "sentence_length_grid": list(DEFAULT_SENTENCE_LENGTH_GRID),
        "sentence_length_bins": list(DEFAULT_SENTENCE_LENGTH_BINS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TIMPA staged sweep with hard-coded staged parameter search: first fix "
            "resteer_steps and search refill_steps over the full trajectory, then "
            "search temperature from the best refill setting, then search sentence "
            "length as a scoring parameter."
        )
    )
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--steer-vectors",
        "--steer_vectors",
        dest="steer_vectors",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--steer-direction",
        "--steer_direction",
        dest="steer_direction",
        type=str,
        choices=["positive", "negative"],
        required=True,
    )
    parser.add_argument(
        "--steer-layers",
        "--steer_layers",
        dest="steer_layers",
        type=str,
        default="25",
        help="Comma-separated layer ids, sandbox.py style. Example: 16,25,31",
    )
    parser.add_argument("--steer-alpha", "--steer_alpha", dest="steer_alpha", type=float, default=500.0)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--max-seq-len", "--max_seq_len", dest="max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--sampling-temp", "--sampling_temp", dest="sampling_temp", type=float, default=1.0)
    parser.add_argument("--identify-temp", "--identify_temp", dest="identify_temp", type=float, default=0.5)
    parser.add_argument("--alpha-decay", "--alpha_decay", dest="alpha_decay", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.device)

    prompts = load_prompts(args.dataset, tokenizer, args.max_seq_len)
    steer_layers = parse_layer_list(args.steer_layers)
    steer_vectors = build_steer_vectors(
        args.steer_vectors,
        args.steer_direction,
        steer_layers,
        args.steer_alpha,
        args.device,
        torch.bfloat16,
    )

    search_plan = get_hardcoded_search_plan()
    refill_grid = list(search_plan["refill_steps_grid"])
    sampling_temp_grid = list(search_plan["sampling_temp_grid"])
    sentence_length_grid = list(search_plan["sentence_length_grid"])
    sentence_length_bins = list(search_plan["sentence_length_bins"])

    raw_dir = Path("results") / "raw" / args.exp_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    pair_results: list[dict[str, Any]] = []
    for refill_steps in tqdm(
        refill_grid,
        desc="Stage 1: fixed resteer_steps, sweep refill_steps",
        unit="refill",
    ):
        pair_result = run_resteer_pair(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            steer_vectors=steer_vectors,
            direction=args.steer_direction,
            resteer_steps=int(search_plan["resteer_steps"]),
            refill_steps=refill_steps,
            batch_size=max(1, args.batch_size),
            max_seq_len=args.max_seq_len,
            sampling_temp=args.sampling_temp,
            identify_temp=args.identify_temp,
            alpha_decay=args.alpha_decay,
            device=args.device,
        )
        pair_results.append(pair_result)

    best_pair_result = max(
        pair_results,
        key=lambda x: float(x["best_step_summary"]["combined_score"]),
    )
    print_stage1_summary(best_pair_result)

    temp_results: list[dict[str, Any]] = []
    best_refill_steps = int(best_pair_result["refill_steps"])
    for sampling_temp in tqdm(
        sampling_temp_grid,
        desc="Stage 2: fixed best refill_steps, sweep sampling_temp",
        unit="temp",
    ):
        temp_result = run_resteer_pair(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            steer_vectors=steer_vectors,
            direction=args.steer_direction,
            resteer_steps=int(search_plan["resteer_steps"]),
            refill_steps=best_refill_steps,
            batch_size=max(1, args.batch_size),
            max_seq_len=args.max_seq_len,
            sampling_temp=float(sampling_temp),
            identify_temp=args.identify_temp,
            alpha_decay=args.alpha_decay,
            device=args.device,
        )
        temp_result["sampling_temp"] = float(sampling_temp)
        temp_results.append(temp_result)

    best_temp_result = max(
        temp_results,
        key=lambda x: float(x["best_step_summary"]["combined_score"]),
    )

    sentence_length_search = evaluate_sentence_length_thresholds(
        best_temp_result,
        sentence_length_grid,
    )
    sentence_length_bin_eval = evaluate_sentence_length_bins(best_temp_result, sentence_length_bins)
    print_stage2_summary(best_temp_result, sentence_length_search, sentence_length_bin_eval)
    heatmap_paths = save_stage_heatmaps(
        exp_name=args.exp_name,
        stage1_results=pair_results,
        stage2_results=temp_results,
        stage3_search=sentence_length_search,
    )

    output = {
        "exp_name": args.exp_name,
        "dataset": str(args.dataset.resolve()),
        "steer_vectors": str(args.steer_vectors.resolve()),
        "steer_direction": args.steer_direction,
        "model": MODEL_ID,
        "seed": args.seed,
        "device": args.device,
        "steer_layers": steer_layers,
        "steer_alpha": args.steer_alpha,
        "sampling_temp": args.sampling_temp,
        "identify_temp": args.identify_temp,
        "alpha_decay": bool(args.alpha_decay),
        "max_seq_len": args.max_seq_len,
        "num_prompts": len(prompts),
        "hardcoded_search_plan": search_plan,
        "selection_metric": "harmonic_mean(target_sentiment, 1 - normalized_perplexity)",
        "artifacts": {
            "json_directory": str(raw_dir.resolve()),
            "image_directory": str((Path("results") / "image" / args.exp_name).resolve()),
            "heatmaps": heatmap_paths,
        },
        "stage1": {
            "search_space": {
                "resteer_steps": int(search_plan["resteer_steps"]),
                "refill_steps_grid": refill_grid,
            },
            "matrix": build_stage1_matrix(pair_results),
            "pair_results": pair_results,
            "best_pair_result": best_pair_result,
        },
        "stage2": {
            "search_space": {
                "resteer_steps": int(search_plan["resteer_steps"]),
                "refill_steps": best_refill_steps,
                "sampling_temp_grid": sampling_temp_grid,
            },
            "matrix": build_stage2_matrix(temp_results),
            "temp_results": temp_results,
            "best_temp_result": best_temp_result,
        },
        "stage3": {
            "sentence_length_grid": sentence_length_grid,
            "sentence_length_bins": sentence_length_bins,
            "sentence_length_search": sentence_length_search,
            "sentence_length_bin_evaluation": sentence_length_bin_eval,
        },
    }

    out_path = raw_dir / "timpa_staged_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
