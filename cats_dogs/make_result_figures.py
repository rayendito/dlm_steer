#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("matplotlib is required for PDF figures; install it in the experiment env") from exc


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def save_bar_pdf(path: Path, title: str, labels: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.9), 4.6))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_line_pdf(path: Path, title: str, xs: list[float], series: dict[str, list[float]], xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, vals in series.items():
        ax.plot(xs, vals, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_by(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(group_key, ""))
        if not key:
            continue
        prev = grouped.get(key)
        if prev is None or to_float(row, "harmonic_score") > to_float(prev, "harmonic_score"):
            grouped[key] = row
    return list(grouped.values())


def vector_count_figure(cats_dir: Path, out_dir: Path) -> None:
    rows = read_csv(cats_dir / "vector_count_summary.csv")
    rows = [r for r in rows if r.get("status") == "complete" or r.get("num_records")]
    if not rows:
        return
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            n = int(float(row.get("num_samples", row.get("n", ""))))
        except ValueError:
            continue
        grouped.setdefault(n, []).append(row)
    xs, target, harmonic, ppl = [], [], [], []
    for n in sorted(grouped):
        subset = grouped[n]
        xs.append(n)
        target.append(float(np.mean([to_float(r, "target_prob") for r in subset])))
        harmonic.append(float(np.mean([to_float(r, "harmonic_score") for r in subset])))
        finite = [to_float(r, "perplexity", math.inf) for r in subset if math.isfinite(to_float(r, "perplexity", math.inf))]
        ppl.append(float(np.mean(finite)) if finite else math.nan)
    save_line_pdf(
        out_dir / "vector_count_ablation.pdf",
        "Cats/Dogs Steering Vector Count",
        xs,
        {"target probability": target, "harmonic score": harmonic},
        "sentences per concept used for vector",
        "score",
    )
    save_line_pdf(
        out_dir / "vector_count_perplexity.pdf",
        "Cats/Dogs Vector Count Perplexity",
        xs,
        {"perplexity": ppl},
        "sentences per concept used for vector",
        "Qwen perplexity",
    )


def hyperparam_figures(cats_dir: Path, out_dir: Path) -> None:
    for stage in ["asap_refill", "asap_sampling_temp", "asap_identify_temp"]:
        rows = read_csv(cats_dir / stage / "asap_hyperparam_trials.csv")
        if not rows:
            continue
        if stage == "asap_refill":
            key = "refill_steps"
            title = "Refill Steps Ablation"
        elif stage == "asap_sampling_temp":
            key = "sampling_temp"
            title = "Sampling Temperature Ablation"
        else:
            key = "identify_temp"
            title = "Identify Temperature Ablation"
        chosen = best_by(rows, key)
        chosen = sorted(chosen, key=lambda r: to_float(r, key))
        labels = [str(r.get(key)) for r in chosen]
        save_bar_pdf(
            out_dir / f"{stage}.pdf",
            title,
            labels,
            {
                "target probability": [to_float(r, "target_prob") for r in chosen],
                "harmonic score": [to_float(r, "harmonic_score") for r in chosen],
            },
            "score",
        )
        save_bar_pdf(
            out_dir / f"{stage}_perplexity.pdf",
            f"{title}: Perplexity",
            labels,
            {"perplexity": [to_float(r, "perplexity") for r in chosen]},
            "Qwen perplexity",
        )


def length_figure(cats_dir: Path, out_dir: Path) -> None:
    best_path = cats_dir / "asap_sampling_temp" / "asap_length_analysis.csv"
    rows = read_csv(best_path)
    if not rows:
        return
    best_overall = max(rows, key=lambda r: to_float(r, "harmonic_score"))
    keys = ["k", "refill_steps", "sampling_temp", "identify_temp"]
    subset = [r for r in rows if all(str(r.get(k)) == str(best_overall.get(k)) for k in keys)]
    order = {"short": 0, "medium": 1, "long": 2}
    subset = sorted(subset, key=lambda r: order.get(str(r.get("length_bin")), 99))
    labels = [str(r.get("length_bin")) for r in subset]
    save_bar_pdf(
        out_dir / "sentence_length_analysis.pdf",
        "Cats/Dogs Effect by Sentence Length",
        labels,
        {
            "target probability": [to_float(r, "target_prob") for r in subset],
            "harmonic score": [to_float(r, "harmonic_score") for r in subset],
        },
        "score",
    )


def baseline_comparison(cats_dir: Path, out_dir: Path) -> None:
    prompt = read_csv(cats_dir / "prompt_baseline" / "cats_dogs_prompt_summary.csv")
    steering_sources = [
        cats_dir / "asap_refill" / "asap_best_config.json",
        cats_dir / "asap_sampling_temp" / "asap_best_config.json",
        cats_dir / "asap_identify_temp" / "asap_best_config.json",
    ]
    steering = []
    for p in steering_sources:
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                row = json.load(f)
            row["method"] = "steering"
            row["direction"] = p.parent.name.replace("asap_", "")
            steering.append(row)
    prompt_overall = [r for r in prompt if r.get("direction") == "overall"]
    rows = steering + prompt_overall
    if not rows:
        return
    labels = [str(r.get("direction", r.get("method", "row"))) for r in rows]
    save_bar_pdf(
        out_dir / "steering_vs_prompting.pdf",
        "Cats/Dogs Steering vs Prompting",
        labels,
        {
            "target probability": [to_float(r, "target_prob") for r in rows],
            "harmonic score": [to_float(r, "harmonic_score") for r in rows],
        },
        "score",
    )
    if prompt:
        labels = [str(r.get("direction")) for r in prompt]
        save_bar_pdf(
            out_dir / "prompting_semantic_similarity.pdf",
            "Prompting Semantic Similarity",
            labels,
            {"semantic similarity": [to_float(r, "semantic_similarity") for r in prompt]},
            "TF-IDF cosine similarity",
        )


def imdb_figure(cats_dir: Path, out_dir: Path) -> None:
    rows = read_csv(cats_dir / "imdb_prompt_baseline" / "imdb_prompt_summary.csv")
    if not rows:
        return
    labels = [str(r.get("direction")) for r in rows]
    save_bar_pdf(
        out_dir / "imdb_prompt_baseline.pdf",
        "IMDB Prompting Baseline",
        labels,
        {
            "target probability": [to_float(r, "target_prob") for r in rows],
            "harmonic score": [to_float(r, "harmonic_score") for r in rows],
            "semantic similarity": [to_float(r, "semantic_similarity") for r in rows],
        },
        "score",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats-dir", type=Path, default=Path("cats_dogs"))
    ap.add_argument("--output-dir", type=Path, default=Path("cats_dogs/figures"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_count_figure(args.cats_dir, args.output_dir)
    hyperparam_figures(args.cats_dir, args.output_dir)
    length_figure(args.cats_dir, args.output_dir)
    baseline_comparison(args.cats_dir, args.output_dir)
    imdb_figure(args.cats_dir, args.output_dir)
    print(f"Wrote figures under {args.output_dir}")


if __name__ == "__main__":
    main()
