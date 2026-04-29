#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            payload = json.load(f)

        params = payload.get("params", {})
        averages = payload.get("averages", {})
        first = averages.get("first_half_metrics", {})
        second = averages.get("second_half_metrics", {})
        full = averages.get("full_text_metrics", {})
        layers = tuple(int(x) for x in params.get("steer_idx", []))

        rows.append(
            {
                "exp_name": str(params.get("exp_name", metrics_path.parent.name)),
                "layer_count": len(layers),
                "layers": ",".join(str(x) for x in layers),
                "steer_direction": str(params.get("steer_direction", "")),
                "first_half_matches_direction": bool(params.get("first_half", True)),
                "steer_alpha": float(params.get("steer_alpha", 0.0)),
                "first_half_scale": float(params.get("first_half_scale", 0.0)),
                "second_half_scale": float(params.get("second_half_scale", 0.0)),
                "block_length": int(params.get("block_length", 0)),
                "batch_size": int(params.get("batch_size", 0)),
                "first_half_positive": float(first.get("positive_sentiment", 0.0)),
                "first_half_negative": float(first.get("negative_sentiment", 0.0)),
                "first_half_perplexity": float(first.get("perplexity", 0.0)),
                "second_half_positive": float(second.get("positive_sentiment", 0.0)),
                "second_half_negative": float(second.get("negative_sentiment", 0.0)),
                "second_half_perplexity": float(second.get("perplexity", 0.0)),
                "full_text_positive": float(full.get("positive_sentiment", 0.0)),
                "full_text_negative": float(full.get("negative_sentiment", 0.0)),
                "full_text_perplexity": float(full.get("perplexity", 0.0)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No metrics.json files found under results directory.")

    fieldnames = [
        "exp_name",
        "layer_count",
        "layers",
        "steer_direction",
        "first_half_matches_direction",
        "steer_alpha",
        "first_half_scale",
        "second_half_scale",
        "block_length",
        "batch_size",
        "first_half_positive",
        "first_half_negative",
        "first_half_perplexity",
        "second_half_positive",
        "second_half_negative",
        "second_half_perplexity",
        "full_text_positive",
        "full_text_negative",
        "full_text_perplexity",
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row["layer_count"],
            row["layers"],
            row["steer_alpha"],
            row["first_half_scale"],
            row["second_half_scale"],
            row["exp_name"],
        ),
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize eval_pattern results into one CSV")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-csv", type=Path, default=Path("results_summary.csv"))
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    write_csv(rows, args.out_csv)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
