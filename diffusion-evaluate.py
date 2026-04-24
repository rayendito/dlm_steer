#!/usr/bin/env python3
"""
Evaluate diffusion steering validation sweep JSON with:
- negative sentiment (from eval_dito.score_labels)
- perplexity (from eval_dito.perplexity)

If multiple prompts are present, heatmaps are created per prompt
(no averaging across prompts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------- parsing sweep json ----------------------------
from dotenv import load_dotenv
load_dotenv()
from eval_dito import score_labels, perplexity

def parse_val_sweep_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "results" not in payload:
        raise ValueError("Invalid sweep JSON: expected object with 'results'.")
    results = payload["results"]
    if not isinstance(results, list):
        raise ValueError("Invalid sweep JSON: 'results' must be a list.")
    # only evaluate steered entries for alpha x layer heatmaps
    return [
        r for r in results
        if isinstance(r, dict)
        and r.get("mode") == "steered"
        and isinstance(r.get("layer"), int)
        and r["layer"] >= 0
    ]


# ------------------------------- eval_dito metrics --------------------------------


# ----------------------------- matrix + heatmap -------------------------------

def build_matrices(
    scores: dict[tuple[int, float], dict[str, float]],
) -> tuple[list[int], list[int], "Any", "Any", "Any"]:
    import numpy as np

    if not scores:
        return [], [], np.array([]), np.array([]), np.array([])
    layers = sorted({k[0] for k in scores.keys()})
    alphas = sorted({k[1] for k in scores.keys()})
    layer_to_i = {l: i for i, l in enumerate(layers)}
    alpha_to_i = {a: i for i, a in enumerate(alphas)}

    M_neg = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    M_ppl = np.full((len(layers), len(alphas)), np.nan, dtype=float)
    for (layer, alpha), s in scores.items():
        li = layer_to_i[layer]
        ai = alpha_to_i[alpha]
        M_neg[li, ai] = s["negative_sentiment"]
        M_ppl[li, ai] = s["perplexity"]
    M_mul = M_neg * M_ppl
    return layers, alphas, M_neg, M_ppl, M_mul


def _draw_pair(
    axes: list[Any],
    layers: list[int],
    alphas: list[int],
    M_neg: Any,
    M_ppl: Any,
    M_mul: Any,
    row_title: str,
) -> None:
    import numpy as np

    def _fmt_alpha(a: float) -> str:
        # Keep labels compact/readable (avoid long float artifacts)
        if abs(a - round(a)) < 1e-9:
            return str(int(round(a)))
        return f"{a:.3g}"

    def _robust_normalize(M: Any, lo_pct: float = 5.0, hi_pct: float = 95.0) -> Any:
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

    # Remove perplexity outlier impact for visualization clarity:
    # clip to robust percentile range, then min-max normalize.
    # Keep raw perplexity values unchanged in saved JSON/results.
    M_ppl_plot = _robust_normalize(M_ppl)
    # High-is-better combined score:
    # - high negative sentiment is good
    # - low perplexity is good -> convert using (1 - normalized perplexity)
    # Score range is [0, 1], with values near 1 being best.
    M_combo = M_neg * (1.0 - M_ppl_plot)

    fig = axes[0].figure
    for ax, M, title, cmap, vmin, vmax, cbl in [
        (axes[0], M_neg, f"{row_title} - Negative sentiment", "magma", 0.0, 1.0, "score"),
        (axes[1], M_ppl_plot, f"{row_title} - Perplexity (normalized)", "viridis_r", 0.0, 1.0, "normalized ppl"),
        (axes[2], M_combo, f"{row_title} - Combined score (higher is better)", "cividis", 0.0, 1.0, "score"),
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
            # Show a subset of ticks when many alphas to avoid overlap
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


def plot_heatmaps_per_prompt(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]],
    out_path: Path,
    show: bool = False,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_ids = sorted(per_prompt_scores.keys())
    n_rows = max(1, len(prompt_ids))
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5 * n_rows), constrained_layout=True)

    if n_rows == 1:
        axes = [axes]

    for i, prompt_idx in enumerate(prompt_ids):
        layers, alphas, M_neg, M_ppl, M_mul = build_matrices(per_prompt_scores[prompt_idx])
        _draw_pair(
            axes=axes[i],
            layers=layers,
            alphas=alphas,
            M_neg=M_neg,
            M_ppl=M_ppl,
            M_mul=M_mul,
            row_title=f"Prompt {prompt_idx}",
        )

    fig.suptitle("Diffusion steering sweep per prompt: negative sentiment, perplexity, and product", fontsize=12)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved heatmap: {out_path}", file=sys.stderr)


def build_avg_scores(
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]]
) -> dict[tuple[int, float], dict[str, float]]:
    from collections import defaultdict

    agg: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(
        lambda: {"negative_sentiment": [], "perplexity": []}
    )
    for prompt_map in per_prompt_scores.values():
        for key, s in prompt_map.items():
            agg[key]["negative_sentiment"].append(float(s["negative_sentiment"]))
            agg[key]["perplexity"].append(float(s["perplexity"]))

    avg_scores: dict[tuple[int, float], dict[str, float]] = {}
    for key, vals in agg.items():
        avg_scores[key] = {
            "negative_sentiment": sum(vals["negative_sentiment"]) / max(1, len(vals["negative_sentiment"])),
            "perplexity": sum(vals["perplexity"]) / max(1, len(vals["perplexity"])),
        }
    return avg_scores


def print_top5_combinations(avg_scores: dict[tuple[int, float], dict[str, float]]) -> None:
    import numpy as np

    if not avg_scores:
        print("No averaged scores available for top-5 ranking.", file=sys.stderr)
        return

    # Build robust normalized perplexity over averaged cells
    cells = list(avg_scores.items())
    ppl_vals = np.array([float(v["perplexity"]) for _, v in cells], dtype=float)
    finite = np.isfinite(ppl_vals)
    ppl_norm = np.zeros_like(ppl_vals, dtype=float)
    if np.any(finite):
        vals = ppl_vals[finite]
        lo = float(np.nanpercentile(vals, 5))
        hi = float(np.nanpercentile(vals, 95))
        if hi < lo:
            lo, hi = hi, lo
        clipped = np.clip(ppl_vals, lo, hi)
        cmin = float(np.nanmin(clipped[finite]))
        cmax = float(np.nanmax(clipped[finite]))
        if cmax > cmin:
            ppl_norm = (clipped - cmin) / (cmax - cmin)

    ranked = []
    for i, ((layer, alpha), s) in enumerate(cells):
        neg = float(s["negative_sentiment"])
        ppl = float(s["perplexity"])
        combo = neg * (1.0 - float(ppl_norm[i]))
        ranked.append((combo, layer, alpha, neg, ppl))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # Enforce unique layers in top-k output.
    topk = []
    used_layers = set()
    for item in ranked:
        _, layer, _, _, _ = item
        if layer in used_layers:
            continue
        topk.append(item)
        used_layers.add(layer)
        if len(topk) == 5:
            break

    print("\nTop 5 averaged combinations by combined score (unique layers, higher is better):", file=sys.stderr)
    for rank, (combo, layer, alpha, neg, ppl) in enumerate(topk, start=1):
        print(
            f"{rank}. layer={layer}, alpha={alpha:g}, combined={combo:.4f}, neg={neg:.4f}, ppl={ppl:.4f}",
            file=sys.stderr,
        )


def plot_avg_heatmap(
    avg_scores: dict[tuple[int, float], dict[str, float]],
    out_path: Path,
    show: bool = False,
) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, alphas, M_neg, M_ppl, M_mul = build_matrices(avg_scores)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    _draw_pair(
        axes=list(axes),
        layers=layers,
        alphas=alphas,
        M_neg=M_neg,
        M_ppl=M_ppl,
        M_mul=M_mul,
        row_title="Average (all prompts)",
    )
    fig.suptitle("Diffusion steering sweep average across prompts", fontsize=12)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved avg heatmap: {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation for diffusion-steering-val JSON output")
    ap.add_argument(
        "--sweep-file",
        type=Path,
        default=Path("diffusion_steering_val_sweep.json"),
        help="Output JSON from diffusion-steering-val.py",
    )
    ap.add_argument("--out-json", type=Path, default=Path("diffusion_eval_sweep.json"))
    ap.add_argument("--out-png", type=Path, default=Path("diffusion_eval_sweep_heatmaps.png"))
    ap.add_argument(
        "--out-png-avg",
        type=Path,
        default=Path("diffusion_eval_sweep_heatmaps_avg.png"),
        help="Output PNG for average over all prompts (same 3-column style)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only score first N steered rows (0=all)")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for eval_dito scoring (reduce if you hit OOM)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window with the heatmaps (use on machine with display)",
    )
    args = ap.parse_args()

    if not args.sweep_file.is_file():
        print(f"Missing sweep file: {args.sweep_file}", file=sys.stderr)
        sys.exit(1)

    rows = parse_val_sweep_json(args.sweep_file)
    if not rows:
        print("No steered rows found in sweep JSON.", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    if args.batch_size <= 0:
        print("--batch-size must be > 0", file=sys.stderr)
        sys.exit(1)

    per_sample_scores: dict[tuple[int, int, float], dict[str, float]] = {}
    total = len(rows)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_rows = rows[start:end]
        texts = [str(r.get("generation", "")) for r in batch_rows]
        try:
            _, _, p_neg = score_labels(texts)
            ppls = perplexity(texts)
        except Exception as e:
            print(f"Metric eval failed in batch [{start}:{end}]: {e}", file=sys.stderr)
            raise

        for i, row in enumerate(batch_rows):
            prompt_idx = int(row.get("prompt_idx", -1))
            layer = int(row["layer"])
            alpha = float(row["alpha"])
            s = {
                "negative_sentiment": float(p_neg[i].item()),
                "perplexity": float(ppls[i].item()),
            }
            per_sample_scores[(prompt_idx, layer, alpha)] = s
            print(
                f"Scored prompt={prompt_idx} layer={layer} alpha={alpha}  neg={s['negative_sentiment']:.4f}  ppl={s['perplexity']:.4f}",
                file=sys.stderr,
            )
        print(f"Finished batch {start}-{end - 1} / {total - 1}", file=sys.stderr)

    # Replace NaN/inf perplexity with global max finite perplexity.
    finite_ppls = [
        s["perplexity"]
        for s in per_sample_scores.values()
        if s["perplexity"] == s["perplexity"] and s["perplexity"] != float("inf") and s["perplexity"] != float("-inf")
    ]
    max_finite_ppl = max(finite_ppls) if finite_ppls else 1.0
    for s in per_sample_scores.values():
        ppl = s["perplexity"]
        if ppl != ppl or ppl == float("inf") or ppl == float("-inf"):
            s["perplexity"] = max_finite_ppl

    # group by prompt for per-prompt heatmaps
    per_prompt_scores: dict[int, dict[tuple[int, float], dict[str, float]]] = {}
    for (prompt_idx, layer, alpha), s in per_sample_scores.items():
        prompt_map = per_prompt_scores.setdefault(prompt_idx, {})
        prompt_map[(layer, alpha)] = s

    out_payload = {
        "sweep_file": str(args.sweep_file),
        "scorer": "eval_dito.score_labels + eval_dito.perplexity",
        "results_per_sample": [
            {
                "prompt_idx": prompt_idx,
                "layer": layer,
                "alpha": alpha,
                "negative_sentiment": per_sample_scores[(prompt_idx, layer, alpha)]["negative_sentiment"],
                "perplexity": per_sample_scores[(prompt_idx, layer, alpha)]["perplexity"],
            }
            for (prompt_idx, layer, alpha) in sorted(per_sample_scores.keys())
        ],
        "results_by_prompt_layer_alpha": [
            {
                "prompt_idx": prompt_idx,
                "layer": layer,
                "alpha": alpha,
                "negative_sentiment": s["negative_sentiment"],
                "perplexity": s["perplexity"],
            }
            for prompt_idx in sorted(per_prompt_scores.keys())
            for (layer, alpha), s in sorted(per_prompt_scores[prompt_idx].items())
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"Wrote {args.out_json}", file=sys.stderr)

    plot_heatmaps_per_prompt(per_prompt_scores, args.out_png, show=args.show)
    avg_scores = build_avg_scores(per_prompt_scores)
    print_top5_combinations(avg_scores)
    plot_avg_heatmap(avg_scores, args.out_png_avg, show=args.show)


if __name__ == "__main__":
    main()
