#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict[str, float | str | int | bool]]:
    rows: list[dict[str, float | str | int | bool]] = []
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
                }
            )
    return rows


def plot(rows: list[dict[str, float | str | int | bool]], out_png: Path, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise ValueError("No rows found in CSV.")

    ppls = np.array([float(row["second_half_perplexity"]) for row in rows], dtype=float)
    inv_ppl = 1.0 / np.maximum(ppls, 1e-8)
    inv_min = float(inv_ppl.min())
    inv_max = float(inv_ppl.max())
    if inv_max > inv_min:
        inv_norm = (inv_ppl - inv_min) / (inv_max - inv_min)
    else:
        inv_norm = np.ones_like(inv_ppl)

    for i, row in enumerate(rows):
        row["inv_second_half_perplexity"] = float(inv_ppl[i])
        row["combined_score"] = float(
            float(row["first_half_positive"]) * float(row["second_half_negative"]) * inv_norm[i]
        )

    groups: dict[str, list[dict[str, float | str | int | bool]]] = {}
    for row in rows:
        groups.setdefault(str(row["layers"]), []).append(row)

    ordered_keys = sorted(groups.keys(), key=lambda key: (len(key.split(",")), key))
    fig, axes = plt.subplots(1, len(ordered_keys), figsize=(6 * len(ordered_keys), 5), constrained_layout=True)
    if len(ordered_keys) == 1:
        axes = [axes]

    all_first_pos = [float(row["first_half_positive"]) for row in rows]
    vmin = min(all_first_pos)
    vmax = max(all_first_pos)

    last_scatter = None
    for ax, key in zip(axes, ordered_keys):
        items = groups[key]
        x = [float(row["second_half_negative"]) for row in items]
        y = [float(row["inv_second_half_perplexity"]) for row in items]
        c = [float(row["first_half_positive"]) for row in items]
        s = [90 + 35 * float(row["steer_alpha"]) for row in items]

        last_scatter = ax.scatter(
            x,
            y,
            c=c,
            s=s,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )

        for row in items:
            ax.text(
                float(row["second_half_negative"]) + 0.003,
                float(row["inv_second_half_perplexity"]) + 0.0005,
                f"a{row['steer_alpha']:g}\nfh{row['first_half_scale']:g}/sh{row['second_half_scale']:g}",
                fontsize=7,
                alpha=0.8,
            )

        best_item = max(items, key=lambda row: float(row["combined_score"]))
        ax.scatter(
            [float(best_item["second_half_negative"])],
            [float(best_item["inv_second_half_perplexity"])],
            s=320,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            marker="o",
        )

        ax.set_title(f"Layers {key}")
        ax.set_xlabel("second_half_negative")
        ax.set_ylabel("inverse second_half_perplexity")
        ax.grid(alpha=0.25)

    if last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes, fraction=0.03, pad=0.02, label="first_half_positive")

    fig.suptitle(
        "Eval Pattern Tradeoff by Layer Usage\nx = second-half negative, y = inverse perplexity, color = first-half positive",
        fontsize=13,
    )
    fig.savefig(out_png, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot eval_pattern tradeoff heatmaps by layer usage")
    ap.add_argument("--csv", type=Path, default=Path("results_summary.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("eval_pattern_tradeoff.png"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot(rows, args.out_png, show=args.show)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
