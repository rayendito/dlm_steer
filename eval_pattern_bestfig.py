#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "exp_name": row["exp_name"],
                    "layer_count": int(row["layer_count"]),
                    "layers": row["layers"],
                    "steer_alpha": float(row["steer_alpha"]),
                    "first_half_scale": float(row["first_half_scale"]),
                    "second_half_scale": float(row["second_half_scale"]),
                    "first_half_positive": float(row["first_half_positive"]),
                    "second_half_negative": float(row["second_half_negative"]),
                    "second_half_perplexity": float(row["second_half_perplexity"]),
                    "score_avg": float(row["score_avg"]) if "score_avg" in row and row["score_avg"] else 0.0,
                }
            )
    return rows


def plot(rows: list[dict[str, float | str | int]], out_png: Path, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise ValueError("No rows to plot.")

    groups: dict[str, list[dict[str, float | str | int]]] = {}
    for row in rows:
        groups.setdefault(str(row["layers"]), []).append(row)
    ordered_keys = sorted(groups.keys(), key=lambda key: (len(key.split(",")), key))

    ppl_vals = np.array([float(row["second_half_perplexity"]) for row in rows], dtype=float)
    inv_ppl = 1.0 / np.maximum(ppl_vals, 1e-8)
    cmin = float(inv_ppl.min())
    cmax = float(inv_ppl.max())

    fig, axes = plt.subplots(1, len(ordered_keys), figsize=(6 * len(ordered_keys), 5.5), constrained_layout=True)
    if len(ordered_keys) == 1:
        axes = [axes]

    scatter = None
    for ax, key in zip(axes, ordered_keys):
        items = groups[key]
        x = [float(row["first_half_positive"]) for row in items]
        y = [float(row["second_half_negative"]) for row in items]
        c = [1.0 / max(float(row["second_half_perplexity"]), 1e-8) for row in items]
        s = [140 + 220 * float(row["score_avg"]) for row in items]

        scatter = ax.scatter(
            x,
            y,
            c=c,
            s=s,
            cmap="cividis",
            vmin=cmin,
            vmax=cmax,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
        )

        best = max(items, key=lambda row: float(row["score_avg"]))
        ax.scatter(
            [float(best["first_half_positive"])],
            [float(best["second_half_negative"])],
            s=420,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
        )
        ax.annotate(
            f"best\na{best['steer_alpha']:g}\nfh{best['first_half_scale']:g}/sh{best['second_half_scale']:g}",
            (
                float(best["first_half_positive"]),
                float(best["second_half_negative"]),
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )

        top_items = sorted(items, key=lambda row: float(row["score_avg"]), reverse=True)[:5]
        for row in top_items:
            ax.text(
                float(row["first_half_positive"]) + 0.0015,
                float(row["second_half_negative"]) + 0.0015,
                f"a{row['steer_alpha']:g}\n{row['first_half_scale']:g}/{row['second_half_scale']:g}",
                fontsize=7,
                alpha=0.85,
            )

        ax.set_title(f"Layers {key}")
        ax.set_xlabel("first_half_positive")
        ax.set_ylabel("second_half_negative")
        ax.grid(alpha=0.25)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.02, label="inverse second_half_perplexity")

    fig.suptitle(
        "Best Parameter Regions by Layer Usage\nright = stronger first-half positive, up = stronger second-half negative, brighter = lower perplexity",
        fontsize=13,
    )
    fig.savefig(out_png, dpi=170)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot best eval_pattern params by layer usage")
    ap.add_argument("--csv", type=Path, default=Path("results_ranked_balanced.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("eval_pattern_best_params.png"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot(rows, args.out_png, show=args.show)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
