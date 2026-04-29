#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "exp_name": row["exp_name"],
                    "layer_count": int(row["layer_count"]),
                    "layers": row["layers"],
                    "steer_direction": row["steer_direction"],
                    "first_half_matches_direction": row["first_half_matches_direction"] == "True",
                    "steer_alpha": float(row["steer_alpha"]),
                    "first_half_scale": float(row["first_half_scale"]),
                    "second_half_scale": float(row["second_half_scale"]),
                    "first_half_positive": float(row["first_half_positive"]),
                    "second_half_negative": float(row["second_half_negative"]),
                    "second_half_perplexity": float(row["second_half_perplexity"]),
                }
            )
    return rows


def _norm(values: list[float]) -> list[float]:
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return [0.0 for _ in values]
    lo = min(finite_vals)
    hi = max(finite_vals)
    if hi == lo:
        return [1.0 if math.isfinite(v) else 0.0 for v in values]
    return [((v - lo) / (hi - lo)) if math.isfinite(v) else 0.0 for v in values]


def score_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fh_vals = [float(row["first_half_positive"]) for row in rows]
    sh_vals = [float(row["second_half_negative"]) for row in rows]
    ppl_vals = [float(row["second_half_perplexity"]) for row in rows]

    fh_norm = _norm(fh_vals)
    sh_norm = _norm(sh_vals)
    ppl_norm = _norm(ppl_vals)

    scored: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        out = dict(row)
        out["first_half_positive_norm"] = fh_norm[i]
        out["second_half_negative_norm"] = sh_norm[i]
        out["inverse_ppl_norm"] = 1.0 - ppl_norm[i]
        out["score_avg"] = (fh_norm[i] + sh_norm[i] + (1.0 - ppl_norm[i])) / 3.0
        out["score_mul"] = fh_norm[i] * sh_norm[i] * (1.0 - ppl_norm[i])
        scored.append(out)
    return scored


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "exp_name",
        "layer_count",
        "layers",
        "steer_direction",
        "first_half_matches_direction",
        "steer_alpha",
        "first_half_scale",
        "second_half_scale",
        "first_half_positive",
        "second_half_negative",
        "second_half_perplexity",
        "first_half_positive_norm",
        "second_half_negative_norm",
        "inverse_ppl_norm",
        "score_avg",
        "score_mul",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank eval_pattern summary rows by sentiment/perplexity tradeoff")
    ap.add_argument("--csv", type=Path, default=Path("results_summary.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("results_ranked.csv"))
    ap.add_argument(
        "--balanced-out-csv",
        type=Path,
        default=Path("results_ranked_balanced.csv"),
    )
    ap.add_argument(
        "--ppl-cap",
        type=float,
        default=20.0,
        help="Keep only rows with second_half_perplexity <= this cap in balanced output",
    )
    args = ap.parse_args()

    rows = score_rows(load_rows(args.csv))
    rows_sorted = sorted(rows, key=lambda row: float(row["score_avg"]), reverse=True)
    write_csv(rows_sorted, args.out_csv)

    balanced = [
        row for row in rows_sorted
        if math.isfinite(float(row["second_half_perplexity"])) and float(row["second_half_perplexity"]) <= args.ppl_cap
    ]
    write_csv(balanced, args.balanced_out_csv)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.balanced_out_csv}")
    if rows_sorted:
        top = rows_sorted[0]
        print(
            "Top overall:",
            top["exp_name"],
            f"score_avg={float(top['score_avg']):.4f}",
            f"fh_pos={float(top['first_half_positive']):.4f}",
            f"sh_neg={float(top['second_half_negative']):.4f}",
            f"sh_ppl={float(top['second_half_perplexity']):.4f}",
        )
    if balanced:
        topb = balanced[0]
        print(
            "Top balanced:",
            topb["exp_name"],
            f"score_avg={float(topb['score_avg']):.4f}",
            f"fh_pos={float(topb['first_half_positive']):.4f}",
            f"sh_neg={float(topb['second_half_negative']):.4f}",
            f"sh_ppl={float(topb['second_half_perplexity']):.4f}",
        )


if __name__ == "__main__":
    main()
