#!/usr/bin/env python3
"""
Merge positive- and negative-direction eval summaries under ``extract_vectors/``.

Scans for paired folders ``results_pos_{tag}`` and ``results_neg_{tag}`` (any ``tag`` string,
e.g. ``0``, ``1``, ``20``). When both exist and contain ``eval_scores.json``:

- Reads ``top5_rankings`` (prefers ``by_harmonic_mean_unique_layer``; falls back to
  ``by_combined_unique_layer`` / ``combined``).
- If top-5 lists are missing, recomputes direction-wise harmonic mean
  (sentiment vs. 1 − robust-normalized perplexity) from ``results_per_sample``,
  aligned with ``resteer_val_sweep_eval.py``.
- Computes **cross-direction** harmonic mean on the **full** layer×α grid (union of
  both sweeps; missing direction scores padded with 0), sorts by that score, then
  keeps **top 5 with unique layers** (greedy: best cross score per layer, then next
  unseen layer).
- Pairs top-5 rows **by rank** (1 with 1, …) and reports harmonic mean of the two
  listed direction scores when both lists have that rank.
- Writes ``extract_vectors/merged_{tag}/heatmaps_avg_combined.png`` per paired tag
  (3 panels: pos harmonic, neg harmonic, cross harmonic; same layout as sweep avg).

Writes ``extract_vectors/results.json`` (default path; override with ``--out``).

Run from repo root::

    python extract_vectors/merge_eval_results.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

WORST_HARMONIC_SCORE = 0.0

RE_POS = re.compile(r"^results_pos_(.+)$")
RE_NEG = re.compile(r"^results_neg_(.+)$")


def _discover_tags(ev_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    pos: dict[str, Path] = {}
    neg: dict[str, Path] = {}
    if not ev_dir.is_dir():
        return pos, neg
    for p in ev_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        m = RE_POS.match(name)
        if m:
            pos[m.group(1)] = p
            continue
        m = RE_NEG.match(name)
        if m:
            neg[m.group(1)] = p
    return pos, neg


def _sort_tags(tags: set[str]) -> list[str]:
    def key(t: str) -> tuple[int, str]:
        if t.isdigit():
            return (0, f"{int(t):06d}")
        return (1, t)

    return sorted(tags, key=key)


def _robust_ppl_norm(ppls: np.ndarray, lo_pct: float = 0.0, hi_pct: float = 85.0) -> np.ndarray:
    finite = np.isfinite(ppls)
    if not np.any(finite):
        return np.zeros_like(ppls, dtype=float)
    vals = ppls[finite]
    lo = float(np.nanpercentile(vals, lo_pct))
    hi = float(np.nanpercentile(vals, hi_pct))
    if hi < lo:
        lo, hi = hi, lo
    clipped = np.clip(ppls, lo, hi)
    cmin = float(np.nanmin(clipped[finite]))
    cmax = float(np.nanmax(clipped[finite]))
    if cmax > cmin:
        return (clipped - cmin) / (cmax - cmin)
    return np.zeros_like(ppls, dtype=float)


def _harmonic_sent_inv_ppl_scalar(sent: float, ppl_norm: float, eps: float = 1e-8) -> float:
    a = max(float(sent), 0.0)
    b = max(1.0 - float(ppl_norm), 0.0)
    d = a + b
    if d <= eps:
        return 0.0
    return (2.0 * a * b) / d


def _harmonic_two(a: float, b: float, eps: float = 1e-8) -> float:
    a = max(float(a), 0.0)
    b = max(float(b), 0.0)
    d = a + b
    if d <= eps:
        return 0.0
    return (2.0 * a * b) / d


def _grid_axes(
    hm_pos: dict[tuple[int, float], float],
    hm_neg: dict[tuple[int, float], float],
) -> tuple[list[int], list[float]]:
    keys = set(hm_pos.keys()) | set(hm_neg.keys())
    layers = sorted({k[0] for k in keys})
    alphas = sorted({k[1] for k in keys})
    return layers, alphas


def _padded_direction_maps(
    hm_pos: dict[tuple[int, float], float],
    hm_neg: dict[tuple[int, float], float],
) -> tuple[list[int], list[float], dict[tuple[int, float], float], dict[tuple[int, float], float]]:
    layers, alphas = _grid_axes(hm_pos, hm_neg)
    pos_filled: dict[tuple[int, float], float] = {}
    neg_filled: dict[tuple[int, float], float] = {}
    for layer, alpha in product(layers, alphas):
        key = (layer, alpha)
        pos_filled[key] = float(hm_pos.get(key, WORST_HARMONIC_SCORE))
        neg_filled[key] = float(hm_neg.get(key, WORST_HARMONIC_SCORE))
    return layers, alphas, pos_filled, neg_filled


def _cross_harmonic_map(
    pos_filled: dict[tuple[int, float], float],
    neg_filled: dict[tuple[int, float], float],
) -> dict[tuple[int, float], float]:
    return {
        key: float(_harmonic_two(pos_filled[key], neg_filled[key]))
        for key in pos_filled.keys()
    }


def _union_topk_cross_unique_layer(
    hm_pos: dict[tuple[int, float], float],
    hm_neg: dict[tuple[int, float], float],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Cross harmonic on full layer×α grid; missing direction scores use WORST_HARMONIC_SCORE."""
    _, _, pos_filled, neg_filled = _padded_direction_maps(hm_pos, hm_neg)
    cross = _cross_harmonic_map(pos_filled, neg_filled)
    ranked = sorted(
        cross.keys(),
        key=lambda key: (-cross[key], key[0], key[1]),
    )
    out: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for key in ranked:
        layer = int(key[0])
        if layer in seen_layers:
            continue
        seen_layers.add(layer)
        ph = float(pos_filled[key])
        nh = float(neg_filled[key])
        out.append(
            {
                "rank": len(out) + 1,
                "layer": layer,
                "alpha": float(key[1]),
                "pos_harmonic_mean": ph,
                "neg_harmonic_mean": nh,
                "cross_harmonic_mean": float(cross[key]),
            }
        )
        if len(out) >= k:
            break
    return out


def _fmt_alpha(a: float) -> str:
    if abs(a - round(a)) < 1e-9:
        return str(int(round(a)))
    return f"{a:.3g}"


def _matrix_from_map(
    layers: list[int],
    alphas: list[float],
    hm: dict[tuple[int, float], float],
) -> np.ndarray:
    li = {l: i for i, l in enumerate(layers)}
    ai = {a: j for j, a in enumerate(alphas)}
    m = np.full((len(layers), len(alphas)), WORST_HARMONIC_SCORE, dtype=float)
    for (layer, alpha), v in hm.items():
        if layer in li and alpha in ai:
            m[li[layer], ai[alpha]] = float(v)
    return m


def _plot_combined_avg_heatmap(
    tag: str,
    layers: list[int],
    alphas: list[float],
    pos_filled: dict[tuple[int, float], float],
    neg_filled: dict[tuple[int, float], float],
    cross: dict[tuple[int, float], float],
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_pos = _matrix_from_map(layers, alphas, pos_filled)
    m_neg = _matrix_from_map(layers, alphas, neg_filled)
    m_cross = _matrix_from_map(layers, alphas, cross)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    for ax, m, title, cmap in [
        (axes[0], m_pos, f"Tag {tag} — positive harmonic mean", "cividis"),
        (axes[1], m_neg, f"Tag {tag} — negative harmonic mean", "cividis"),
        (axes[2], m_cross, f"Tag {tag} — cross harmonic mean", "cividis"),
    ]:
        im = ax.imshow(
            m,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="harmonic mean")
    fig.suptitle(
        f"Merged val sweep (N={tag}): pos, neg, and cross harmonic means (avg over prompts)",
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved combined avg heatmap: {out_png}", file=sys.stderr)


def _aggregate_per_sample(rows: list[dict[str, Any]], sentiment_key: str) -> dict[tuple[int, float], dict[str, float]]:
    agg: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(
        lambda: {sentiment_key: [], "perplexity": []}
    )
    for r in rows:
        k = (int(r["layer"]), float(r["alpha"]))
        agg[k][sentiment_key].append(float(r[sentiment_key]))
        agg[k]["perplexity"].append(float(r["perplexity"]))
    out: dict[tuple[int, float], dict[str, float]] = {}
    for k, v in agg.items():
        out[k] = {
            sentiment_key: sum(v[sentiment_key]) / max(1, len(v[sentiment_key])),
            "perplexity": sum(v["perplexity"]) / max(1, len(v["perplexity"])),
        }
    return out


def _direction_harmonic_map(
    avg: dict[tuple[int, float], dict[str, float]], sentiment_key: str
) -> dict[tuple[int, float], float]:
    keys = list(avg.keys())
    if not keys:
        return {}
    ppls = np.array([float(avg[k]["perplexity"]) for k in keys], dtype=float)
    ppl_norm = _robust_ppl_norm(ppls)
    hm: dict[tuple[int, float], float] = {}
    for i, k in enumerate(keys):
        sent = float(avg[k][sentiment_key])
        hm[k] = _harmonic_sent_inv_ppl_scalar(sent, float(ppl_norm[i]))
    return hm


def _extract_top5_from_payload(data: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Returns (rows, source_key_used)."""
    tr = data.get("top5_rankings")
    if isinstance(tr, dict):
        if tr.get("by_harmonic_mean_unique_layer"):
            return list(tr["by_harmonic_mean_unique_layer"]), "by_harmonic_mean_unique_layer"
        if tr.get("by_combined_unique_layer"):
            return list(tr["by_combined_unique_layer"]), "by_combined_unique_layer"
    return [], ""


def _row_score(row: dict[str, Any]) -> float:
    if "harmonic_mean" in row:
        return float(row["harmonic_mean"])
    if "combined" in row:
        return float(row["combined"])
    return 0.0


def _recompute_top5_unique_layers(
    rows: list[dict[str, Any]], sentiment_key: str
) -> list[dict[str, Any]]:
    avg = _aggregate_per_sample(rows, sentiment_key)
    keys = list(avg.keys())
    filtered = [
        k
        for k in keys
        if np.isfinite(avg[k]["perplexity"])
        and np.isfinite(avg[k][sentiment_key])
    ]
    if not filtered:
        return []
    ppls = np.array([float(avg[k]["perplexity"]) for k in filtered], dtype=float)
    ppl_norm = _robust_ppl_norm(ppls)
    ranked: list[tuple[float, int, float]] = []
    for i, k in enumerate(filtered):
        layer, alpha = k
        hm = _harmonic_sent_inv_ppl_scalar(float(avg[k][sentiment_key]), float(ppl_norm[i]))
        ranked.append((hm, layer, alpha))
    ranked.sort(key=lambda x: x[0], reverse=True)
    topk: list[tuple[float, int, float]] = []
    seen: set[int] = set()
    for hm, layer, alpha in ranked:
        if layer in seen:
            continue
        seen.add(layer)
        topk.append((hm, layer, alpha))
        if len(topk) == 5:
            break
    out_rows: list[dict[str, Any]] = []
    for rank, (hm, layer, alpha) in enumerate(topk, start=1):
        s = avg[(layer, alpha)]
        out_rows.append(
            {
                "rank": rank,
                "layer": layer,
                "alpha": alpha,
                sentiment_key: float(s[sentiment_key]),
                "perplexity": float(s["perplexity"]),
                "harmonic_mean": float(hm),
            }
        )
    return out_rows


def _load_eval(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _process_pair(
    tag: str,
    pos_dir: Path,
    neg_dir: Path,
    ev_dir: Path,
) -> dict[str, Any] | None:
    pos_eval = pos_dir / "eval_scores.json"
    neg_eval = neg_dir / "eval_scores.json"
    pos_data = _load_eval(pos_eval)
    neg_data = _load_eval(neg_eval)
    if pos_data is None or neg_data is None:
        return None

    pos_key = str(pos_data.get("sentiment_metric") or "positive_sentiment_prob")
    neg_key = str(neg_data.get("sentiment_metric") or "negative_sentiment_prob")
    pos_rows = list(pos_data.get("results_per_sample") or [])
    neg_rows = list(neg_data.get("results_per_sample") or [])

    pos_top5, pos_src = _extract_top5_from_payload(pos_data)
    neg_top5, neg_src = _extract_top5_from_payload(neg_data)

    if not pos_top5 and pos_rows:
        pos_top5 = _recompute_top5_unique_layers(pos_rows, pos_key)
        pos_src = "recomputed_unique_layers"
    if not neg_top5 and neg_rows:
        neg_top5 = _recompute_top5_unique_layers(neg_rows, neg_key)
        neg_src = "recomputed_unique_layers"

    avg_pos = _aggregate_per_sample(pos_rows, pos_key)
    avg_neg = _aggregate_per_sample(neg_rows, neg_key)
    hm_pos = _direction_harmonic_map(avg_pos, pos_key)
    hm_neg = _direction_harmonic_map(avg_neg, neg_key)
    layers, alphas, pos_filled, neg_filled = _padded_direction_maps(hm_pos, hm_neg)
    cross_filled = _cross_harmonic_map(pos_filled, neg_filled)
    cross_top5 = _union_topk_cross_unique_layer(hm_pos, hm_neg, k=5)

    heatmap_path = ev_dir / "heatmaps" / f"avg_combined_{tag}.png"
    _plot_combined_avg_heatmap(
        tag, layers, alphas, pos_filled, neg_filled, cross_filled, heatmap_path
    )

    paired_by_rank: list[dict[str, Any]] = []
    n_pair = min(len(pos_top5), len(neg_top5), 5)
    for i in range(n_pair):
        pr = pos_top5[i]
        nr = neg_top5[i]
        sp = _row_score(pr)
        sn = _row_score(nr)
        paired_by_rank.append(
            {
                "rank": i + 1,
                "between_directions_harmonic_mean": float(_harmonic_two(sp, sn)),
                "pos": pr,
                "neg": nr,
            }
        )

    return {
        "tag": tag,
        "pos_dir": str(pos_dir.resolve()),
        "neg_dir": str(neg_dir.resolve()),
        "pos_eval_scores": str(pos_eval.resolve()),
        "neg_eval_scores": str(neg_eval.resolve()),
        "pos_top5_source": pos_src or "embedded",
        "neg_top5_source": neg_src or "embedded",
        "pos_top5": pos_top5,
        "neg_top5": neg_top5,
        "paired_top5_by_rank_harmonic": paired_by_rank,
        "intersection_top5_by_cross_harmonic": cross_top5,
        "heatmap_combined": str(heatmap_path.resolve()),
    }


def _tag_sort_key(t: str) -> tuple[int, int | str]:
    return (0, int(t)) if t.isdigit() else (1, t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--extract-vectors-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing results_pos_* / results_neg_* (default: this folder).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("extract_vectors/results.json"),
        help="Output JSON path (default: extract_vectors/results.json from cwd).",
    )
    args = ap.parse_args()

    ev = (args.extract_vectors_dir.resolve() / 'results').resolve()
    pos_map, neg_map = _discover_tags(ev)
    pair_tags = sorted(set(pos_map.keys()) & set(neg_map.keys()), key=_tag_sort_key)

    per_tag: dict[str, Any] = {}
    for tag in pair_tags:
        merged = _process_pair(
            tag,
            pos_map[tag],
            neg_map[tag],
            ev,
        )
        if merged is not None:
            per_tag[tag] = merged

    out_obj: dict[str, Any] = {
        "script": "extract_vectors/merge_eval_results.py",
        "extract_vectors_dir": str(ev),
        "tags_pos_only": _sort_tags(set(pos_map.keys()) - set(neg_map.keys())),
        "tags_neg_only": _sort_tags(set(neg_map.keys()) - set(pos_map.keys())),
        "tags_paired": sorted(per_tag.keys(), key=_tag_sort_key),
        "per_tag": per_tag,
    }

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}", file=sys.stderr)
    print(
        f"Paired tags: {out_obj['tags_paired']}; "
        f"pos-only: {out_obj['tags_pos_only']}; neg-only: {out_obj['tags_neg_only']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
