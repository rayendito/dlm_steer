#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"catdog_cat_minus_dog(?:_val)?_n(?P<n>\d+)(?:_s(?P<seed>\d+))?")


def parse_tag(name: str) -> tuple[int | None, int | None]:
    m = TAG_RE.search(name)
    if not m:
        return None, None
    n = int(m.group("n"))
    seed = int(m.group("seed")) if m.group("seed") else None
    return n, seed


def best_hm(eval_payload: dict[str, Any]) -> dict[str, Any]:
    rows = (
        eval_payload.get("top5_rankings", {})
        .get("by_harmonic_mean_unique_layer", [])
    )
    return rows[0] if rows else {}


def target_metric(eval_payload: dict[str, Any]) -> str:
    return (
        eval_payload.get("target_metric")
        or eval_payload.get("sentiment_metric")
        or "target_prob"
    )


def collect(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(results_root.glob("results_*catdog*/eval_scores.json")):
        direction = "positive" if eval_path.parent.name.startswith("results_pos_") else "negative"
        n, seed = parse_tag(eval_path.parent.name)
        with eval_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        metric = target_metric(payload)
        best = best_hm(payload)
        rows.append(
            {
                "result_dir": str(eval_path.parent),
                "direction": direction,
                "target": "cat" if direction == "positive" else "dog",
                "n": n,
                "seed": seed,
                "layer": best.get("layer"),
                "alpha": best.get("alpha"),
                "harmonic_mean": best.get("harmonic_mean"),
                "target_prob": best.get(metric),
                "perplexity": best.get("perplexity"),
                "metric": metric,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "seed",
        "direction",
        "target",
        "layer",
        "alpha",
        "harmonic_mean",
        "target_prob",
        "perplexity",
        "metric",
        "result_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("extract_vectors"))
    ap.add_argument("--out-csv", type=Path, default=Path("cats_dogs/vector_count_summary.csv"))
    args = ap.parse_args()

    rows = collect(args.results_root)
    write_csv(rows, args.out_csv)
    print(f"wrote {args.out_csv} rows={len(rows)}")

    by_n: dict[tuple[int | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault((row["n"], row["direction"]), []).append(row)
    for (n, direction), group in sorted(by_n.items(), key=lambda kv: (kv[0][0] or -1, kv[0][1])):
        vals = [g for g in group if g["harmonic_mean"] is not None]
        if not vals:
            continue
        hm = sum(float(g["harmonic_mean"]) for g in vals) / len(vals)
        tgt = sum(float(g["target_prob"]) for g in vals) / len(vals)
        ppl = sum(float(g["perplexity"]) for g in vals) / len(vals)
        print(
            f"n={n} direction={direction} seeds={len(vals)} "
            f"hm={hm:.4f} target_prob={tgt:.4f} ppl={ppl:.4f}"
        )


if __name__ == "__main__":
    main()
