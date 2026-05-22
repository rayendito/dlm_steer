#!/usr/bin/env python3
"""
Merge paired-direction eval summaries under ``extract_vectors/results/``.

Discovers folder pairs (same ``tag``, both with ``eval_scores.json``):

- IMDB: ``results_pos_{tag}`` + ``results_neg_{tag}``
- cats-dogs: ``results_cats_{tag}`` + ``results_dogs_{tag}``

For each pair: top-5 per direction, cross-direction top-5 on the full layer×α grid,
and ``heatmaps/avg_combined_imdb_{tag}.png`` or ``avg_combined_catdog_{tag}.png``.

Writes ``extract_vectors/results.json`` (default; override with ``--out``).

Run from repo root::

    python extract_vectors/merge_eval_results.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

WORST_HARMONIC_SCORE = 0.0

# (data id, regex side A, regex side B, abbrev A, abbrev B, eval label A, eval label B)
_PAIR_SPECS: tuple[tuple[str, re.Pattern[str], re.Pattern[str], str, str, str, str], ...] = (
    (
        "imdb",
        re.compile(r"^results_pos_(.+)$"),
        re.compile(r"^results_neg_(.+)$"),
        "pos",
        "neg",
        "positive",
        "negative",
    ),
    (
        "cats-dogs",
        re.compile(r"^results_cats_(.+)$"),
        re.compile(r"^results_dogs_(.+)$"),
        "cats",
        "dogs",
        "cat",
        "dog",
    ),
)


@dataclass(frozen=True)
class DiscoveredPair:
    data: str
    tag: str
    side_a_dir: Path
    side_b_dir: Path
    side_a_abbrev: str
    side_b_abbrev: str
    side_a_label: str
    side_b_label: str

    @property
    def merge_key(self) -> str:
        """Key in ``results.json`` ``per_tag`` (imdb keeps bare tag for compatibility)."""
        if self.data == "imdb":
            return self.tag
        return f"catdog-{self.tag}"

    @property
    def heatmap_basename(self) -> str:
        if self.data == "imdb":
            return f"avg_combined_imdb_{self.tag}"
        return f"avg_combined_catdog_{self.tag}"


def _discover_pairs(ev_dir: Path) -> list[DiscoveredPair]:
    if not ev_dir.is_dir():
        return []

    by_spec: list[tuple[str, re.Pattern[str], re.Pattern[str], str, str, str, str, dict[str, Path], dict[str, Path]]] = [
        (*spec, {}, {}) for spec in _PAIR_SPECS
    ]

    for p in ev_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        for i, row in enumerate(by_spec):
            _, re_a, re_b, _, _, _, _, map_a, map_b = row
            m = re_a.match(name)
            if m:
                map_a[m.group(1)] = p
                continue
            m = re_b.match(name)
            if m:
                map_b[m.group(1)] = p

    pairs: list[DiscoveredPair] = []
    for data, re_a, re_b, abbrev_a, abbrev_b, label_a, label_b, map_a, map_b in by_spec:
        for tag in sorted(set(map_a.keys()) & set(map_b.keys()), key=_tag_sort_key):
            pairs.append(
                DiscoveredPair(
                    data=data,
                    tag=tag,
                    side_a_dir=map_a[tag],
                    side_b_dir=map_b[tag],
                    side_a_abbrev=abbrev_a,
                    side_b_abbrev=abbrev_b,
                    side_a_label=label_a,
                    side_b_label=label_b,
                )
            )
    return pairs


def _unpaired_by_data(ev_dir: Path, pairs: list[DiscoveredPair]) -> dict[str, dict[str, list[str]]]:
    paired_a = {(p.data, p.side_a_dir.name) for p in pairs}
    paired_b = {(p.data, p.side_b_dir.name) for p in pairs}
    out: dict[str, dict[str, list[str]]] = {
        spec[0]: {"side_a_only": [], "side_b_only": []} for spec in _PAIR_SPECS
    }

    if not ev_dir.is_dir():
        return out

    for p in ev_dir.iterdir():
        if not p.is_dir():
            continue
        for data, re_a, re_b, _, _, _, _ in _PAIR_SPECS:
            m = re_a.match(p.name)
            if m and (data, p.name) not in paired_a:
                out[data]["side_a_only"].append(m.group(1))
            m = re_b.match(p.name)
            if m and (data, p.name) not in paired_b:
                out[data]["side_b_only"].append(m.group(1))

    for data in out:
        out[data]["side_a_only"] = _sort_tags(set(out[data]["side_a_only"]))
        out[data]["side_b_only"] = _sort_tags(set(out[data]["side_b_only"]))
    return out


def _sort_tags(tags: set[str]) -> list[str]:
    def key(t: str) -> tuple[int, str]:
        if t.isdigit():
            return (0, f"{int(t):06d}")
        return (1, t)

    return sorted(tags, key=key)


def _tag_sort_key(t: str) -> tuple[int, int | str]:
    return (0, int(t)) if t.isdigit() else (1, t)


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
    hm_a: dict[tuple[int, float], float],
    hm_b: dict[tuple[int, float], float],
) -> tuple[list[int], list[float]]:
    keys = set(hm_a.keys()) | set(hm_b.keys())
    layers = sorted({k[0] for k in keys})
    alphas = sorted({k[1] for k in keys})
    return layers, alphas


def _padded_direction_maps(
    hm_a: dict[tuple[int, float], float],
    hm_b: dict[tuple[int, float], float],
) -> tuple[list[int], list[float], dict[tuple[int, float], float], dict[tuple[int, float], float]]:
    layers, alphas = _grid_axes(hm_a, hm_b)
    filled_a: dict[tuple[int, float], float] = {}
    filled_b: dict[tuple[int, float], float] = {}
    for layer, alpha in product(layers, alphas):
        key = (layer, alpha)
        filled_a[key] = float(hm_a.get(key, WORST_HARMONIC_SCORE))
        filled_b[key] = float(hm_b.get(key, WORST_HARMONIC_SCORE))
    return layers, alphas, filled_a, filled_b


def _cross_harmonic_map(
    filled_a: dict[tuple[int, float], float],
    filled_b: dict[tuple[int, float], float],
) -> dict[tuple[int, float], float]:
    return {
        key: float(_harmonic_two(filled_a[key], filled_b[key]))
        for key in filled_a.keys()
    }


def _union_topk_cross_unique_layer(
    hm_a: dict[tuple[int, float], float],
    hm_b: dict[tuple[int, float], float],
    pair: DiscoveredPair,
    k: int = 5,
) -> list[dict[str, Any]]:
    _, _, filled_a, filled_b = _padded_direction_maps(hm_a, hm_b)
    cross = _cross_harmonic_map(filled_a, filled_b)
    ranked = sorted(cross.keys(), key=lambda key: (-cross[key], key[0], key[1]))
    out: list[dict[str, Any]] = []
    seen_layers: set[int] = set()
    for key in ranked:
        layer = int(key[0])
        if layer in seen_layers:
            continue
        seen_layers.add(layer)
        ah = float(filled_a[key])
        bh = float(filled_b[key])
        row: dict[str, Any] = {
            "rank": len(out) + 1,
            "layer": layer,
            "alpha": float(key[1]),
            "side_a_harmonic_mean": ah,
            "side_b_harmonic_mean": bh,
            "cross_harmonic_mean": float(cross[key]),
            "side_a_label": pair.side_a_label,
            "side_b_label": pair.side_b_label,
        }
        if pair.data == "imdb":
            row["pos_harmonic_mean"] = ah
            row["neg_harmonic_mean"] = bh
        else:
            row["cats_harmonic_mean"] = ah
            row["dogs_harmonic_mean"] = bh
        out.append(row)
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
    pair: DiscoveredPair,
    layers: list[int],
    alphas: list[float],
    filled_a: dict[tuple[int, float], float],
    filled_b: dict[tuple[int, float], float],
    cross: dict[tuple[int, float], float],
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_a = _matrix_from_map(layers, alphas, filled_a)
    m_b = _matrix_from_map(layers, alphas, filled_b)
    m_cross = _matrix_from_map(layers, alphas, cross)
    title_a = f"{pair.side_a_label} harmonic mean"
    title_b = f"{pair.side_b_label} harmonic mean"

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    for ax, m, title, cmap in [
        (axes[0], m_a, f"Tag {pair.tag} — {title_a}", "cividis"),
        (axes[1], m_b, f"Tag {pair.tag} — {title_b}", "cividis"),
        (axes[2], m_cross, f"Tag {pair.tag} — cross harmonic mean", "cividis"),
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
        f"Merged {pair.data} (N={pair.tag}): {pair.side_a_label}, {pair.side_b_label}, cross",
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


def _default_sentiment_key(data: dict[str, Any], fallback: str) -> str:
    return str(data.get("sentiment_metric") or fallback)


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


def _process_pair(pair: DiscoveredPair, ev_dir: Path) -> dict[str, Any] | None:
    side_a_eval = pair.side_a_dir / "eval_scores.json"
    side_b_eval = pair.side_b_dir / "eval_scores.json"
    side_a_data = _load_eval(side_a_eval)
    side_b_data = _load_eval(side_b_eval)
    if side_a_data is None or side_b_data is None:
        return None

    fallback_a = (
        "positive_sentiment_prob"
        if pair.data == "imdb"
        else "cat_sentiment_prob"
    )
    fallback_b = (
        "negative_sentiment_prob"
        if pair.data == "imdb"
        else "dog_sentiment_prob"
    )
    key_a = _default_sentiment_key(side_a_data, fallback_a)
    key_b = _default_sentiment_key(side_b_data, fallback_b)
    rows_a = list(side_a_data.get("results_per_sample") or [])
    rows_b = list(side_b_data.get("results_per_sample") or [])

    top5_a, src_a = _extract_top5_from_payload(side_a_data)
    top5_b, src_b = _extract_top5_from_payload(side_b_data)

    if not top5_a and rows_a:
        top5_a = _recompute_top5_unique_layers(rows_a, key_a)
        src_a = "recomputed_unique_layers"
    if not top5_b and rows_b:
        top5_b = _recompute_top5_unique_layers(rows_b, key_b)
        src_b = "recomputed_unique_layers"

    avg_a = _aggregate_per_sample(rows_a, key_a)
    avg_b = _aggregate_per_sample(rows_b, key_b)
    hm_a = _direction_harmonic_map(avg_a, key_a)
    hm_b = _direction_harmonic_map(avg_b, key_b)
    layers, alphas, filled_a, filled_b = _padded_direction_maps(hm_a, hm_b)
    cross_filled = _cross_harmonic_map(filled_a, filled_b)
    cross_top5 = _union_topk_cross_unique_layer(hm_a, hm_b, pair, k=5)

    heatmap_path = ev_dir / "heatmaps" / f"{pair.heatmap_basename}.png"
    _plot_combined_avg_heatmap(pair, layers, alphas, filled_a, filled_b, cross_filled, heatmap_path)

    paired_by_rank: list[dict[str, Any]] = []
    n_pair = min(len(top5_a), len(top5_b), 5)
    for i in range(n_pair):
        ar = top5_a[i]
        br = top5_b[i]
        paired_by_rank.append(
            {
                "rank": i + 1,
                "between_directions_harmonic_mean": float(_harmonic_two(_row_score(ar), _row_score(br))),
                "side_a": ar,
                "side_b": br,
                "pos": ar,
                "neg": br,
            }
        )

    return {
        "merge_key": pair.merge_key,
        "tag": pair.tag,
        "data": pair.data,
        "side_a_label": pair.side_a_label,
        "side_b_label": pair.side_b_label,
        "side_a_abbrev": pair.side_a_abbrev,
        "side_b_abbrev": pair.side_b_abbrev,
        "side_a_dir": str(pair.side_a_dir.resolve()),
        "side_b_dir": str(pair.side_b_dir.resolve()),
        "side_a_eval_scores": str(side_a_eval.resolve()),
        "side_b_eval_scores": str(side_b_eval.resolve()),
        "side_a_top5_source": src_a or "embedded",
        "side_b_top5_source": src_b or "embedded",
        "side_a_top5": top5_a,
        "side_b_top5": top5_b,
        "pos_dir": str(pair.side_a_dir.resolve()),
        "neg_dir": str(pair.side_b_dir.resolve()),
        "pos_eval_scores": str(side_a_eval.resolve()),
        "neg_eval_scores": str(side_b_eval.resolve()),
        "pos_top5_source": src_a or "embedded",
        "neg_top5_source": src_b or "embedded",
        "pos_top5": top5_a,
        "neg_top5": top5_b,
        "paired_top5_by_rank_harmonic": paired_by_rank,
        "intersection_top5_by_cross_harmonic": cross_top5,
        "heatmap_combined": str(heatmap_path.resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--extract-vectors-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="extract_vectors/ (results live under extract_vectors/results/).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("extract_vectors/results.json"),
        help="Output JSON path (default: extract_vectors/results.json from cwd).",
    )
    args = ap.parse_args()

    ev = (args.extract_vectors_dir.resolve() / "results").resolve()
    pairs = _discover_pairs(ev)
    unpaired = _unpaired_by_data(ev, pairs)

    per_tag: dict[str, Any] = {}
    for pair in pairs:
        merged = _process_pair(pair, ev)
        if merged is not None:
            per_tag[pair.merge_key] = merged

    paired_keys = sorted(per_tag.keys(), key=lambda k: (0, int(k.split("-")[-1])) if k.split("-")[-1].isdigit() else (1, k))

    out_obj: dict[str, Any] = {
        "script": "extract_vectors/merge_eval_results.py",
        "extract_vectors_dir": str(ev),
        "pair_patterns": [
            {
                "data": spec[0],
                "side_a": f"results_{spec[3]}_{{tag}}",
                "side_b": f"results_{spec[4]}_{{tag}}",
            }
            for spec in _PAIR_SPECS
        ],
        "unpaired": unpaired,
        "tags_paired": paired_keys,
        "per_tag": per_tag,
    }

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"Paired merge keys: {paired_keys}", file=sys.stderr)
    for data, u in unpaired.items():
        if u["side_a_only"] or u["side_b_only"]:
            print(
                f"  {data} unpaired — side_a_only={u['side_a_only']}, side_b_only={u['side_b_only']}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
