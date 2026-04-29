#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
                "layers": layers,
                "layer_label": ",".join(str(x) for x in layers) if layers else "unknown",
                "alpha": float(params.get("steer_alpha", 0.0)),
                "first_half_scale": float(params.get("first_half_scale", 0.0)),
                "second_half_scale": float(params.get("second_half_scale", 0.0)),
                "first_pos": float(first.get("positive_sentiment", 0.0)),
                "first_neg": float(first.get("negative_sentiment", 0.0)),
                "first_ppl": float(first.get("perplexity", 0.0)),
                "second_pos": float(second.get("positive_sentiment", 0.0)),
                "second_neg": float(second.get("negative_sentiment", 0.0)),
                "second_ppl": float(second.get("perplexity", 0.0)),
                "full_pos": float(full.get("positive_sentiment", 0.0)),
                "full_neg": float(full.get("negative_sentiment", 0.0)),
                "full_ppl": float(full.get("perplexity", 0.0)),
            }
        )
    return rows


def _group_key(row: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    return (len(row["layers"]), row["layers"])


def plot(rows: list[dict[str, Any]], out_path: Path, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No metrics.json files found to plot.")

    rows = sorted(rows, key=_group_key)
    groups: dict[tuple[int, tuple[int, ...]], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_fp, ax_sp, ax_fn, ax_ppl = axes.flat

    colors = {
        1: "#1b9e77",
        3: "#d95f02",
        5: "#7570b3",
    }

    for (count, layers), items in groups.items():
        label = f"{count} layer" if count == 1 else f"{count} layers"
        label = f"{label}: {','.join(str(x) for x in layers)}"
        color = colors.get(count, "#444444")
        x = [row["alpha"] for row in items]

        ax_fp.scatter(x, [row["first_pos"] for row in items], label=label, color=color, alpha=0.8)
        ax_sp.scatter(x, [row["second_pos"] for row in items], label=label, color=color, alpha=0.8)
        ax_fn.scatter(x, [row["first_neg"] for row in items], color=color, alpha=0.8, marker="o")
        ax_fn.scatter(x, [row["second_neg"] for row in items], color=color, alpha=0.8, marker="x")
        ax_ppl.scatter(x, [row["first_ppl"] for row in items], color=color, alpha=0.8, marker="o")
        ax_ppl.scatter(x, [row["second_ppl"] for row in items], color=color, alpha=0.8, marker="x")

    ax_fp.set_title("First Half Positive Sentiment")
    ax_sp.set_title("Second Half Positive Sentiment")
    ax_fn.set_title("Negative Sentiment: first=o, second=x")
    ax_ppl.set_title("Perplexity: first=o, second=x")

    for ax in (ax_fp, ax_sp, ax_fn, ax_ppl):
        ax.set_xlabel("steer_alpha")
        ax.grid(alpha=0.25)

    ax_fp.set_ylabel("positive score")
    ax_sp.set_ylabel("positive score")
    ax_fn.set_ylabel("negative score")
    ax_ppl.set_ylabel("perplexity")

    handles, labels = ax_fp.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.suptitle("Eval Pattern Overview", fontsize=14)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize eval_pattern metrics.json outputs")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-png", type=Path, default=Path("eval_pattern_overview.png"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    plot(rows, args.out_png, show=args.show)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
