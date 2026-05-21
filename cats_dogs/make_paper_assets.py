#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for paper figures") from exc


COLORS = {
    "target": "#0072B2",
    "harmonic": "#009E73",
    "ppl": "#D55E00",
    "prompt": "#CC79A7",
    "steer": "#56B4E9",
    "gray": "#666666",
}

STAGE_SPECS = [
    ("Refill sweep", "asap_refill", "refill_steps", "Refill steps $u$"),
    ("Sampling-temp sweep", "asap_sampling_temp", "sampling_temp", "Sampling temperature"),
    ("Identify-temp sweep", "asap_identify_temp", "identify_temp", "Identify temperature"),
]
STAGE_TO_DIR = {spec[1].replace("asap_", ""): spec[1] for spec in STAGE_SPECS}
LENGTH_BIN_ORDER = {"short": 0, "medium": 1, "long": 2}
TABLE_METRIC_KEYS = ["Target prob ↑", "PPL ↓", "Harmonic ↑"]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isinf(val):
        return "inf"
    return f"{val:.{digits}f}"


def table_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table_write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(s: Any) -> str:
    out = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        out = out.replace(a, b)
    return out


def table_write_tex(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(keys) + "}",
        r"\toprule",
        " & ".join(latex_escape(k) for k in keys) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(k, "")) for k in keys) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_bundle(out_dir: Path, name: str, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    table_write_csv(out_dir / f"{name}.csv", rows)
    table_write_md(out_dir / f"{name}.md", rows)
    table_write_tex(out_dir / f"{name}.tex", rows, caption, label)


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def group_best(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        k = str(row[key])
        if k not in best or f(row, "harmonic_score") > f(best[k], "harmonic_score"):
            best[k] = row
    return sorted(best.values(), key=lambda r: f(r, key))


def stage_trial_path(cats_dir: Path, stage_dir: str) -> Path:
    return cats_dir / stage_dir / "asap_hyperparam_trials.csv"


def stage_length_path(cats_dir: Path, stage_dir: str) -> Path:
    return cats_dir / stage_dir / "asap_length_analysis.csv"


def compact_steering_row(label: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Stage": label,
        "k": row["k"],
        "u": row["refill_steps"],
        "sampling temp": row["sampling_temp"],
        "identify temp": row["identify_temp"],
        "Target prob ↑": fmt(row["target_prob"]),
        "PPL ↓": fmt(row["perplexity"], 2),
        "Harmonic ↑": fmt(row["harmonic_score"]),
        "N": row["num_records"],
    }


def setup_table(cats_dir: Path, out_dir: Path) -> None:
    prompt = read_csv(cats_dir / "prompt_baseline/cats_dogs_prompt_summary.csv")
    overall = next(r for r in prompt if r["direction"] == "overall")
    rows = [
        {"Decision": "Evaluation split", "Choice": "benchmarks/cats_dogs/train.csv", "Reason": "Matches steering sweep eval set"},
        {"Decision": "Evaluation size", "Choice": str(overall["num_records"]), "Reason": "All available train rows; apples-to-apples baseline"},
        {"Decision": "Steering vector data", "Choice": "n=10 validation examples/class", "Reason": "Cheap vector setting used by final sweeps"},
        {"Decision": "Steering layer / alpha", "Choice": "layer 32, alpha 100", "Reason": "Best available cats/dogs steering setup"},
        {"Decision": "Ablation variables", "Choice": "k, refill u, sampling temp, identify temp", "Reason": "Greedy search under compute limits"},
        {"Decision": "Sentence length", "Choice": "short/medium/long 1/3 quantile bins", "Reason": "Analysis only, not optimized"},
        {"Decision": "Classifier score", "Choice": "Qwen raw next-token P(cat/dog)", "Reason": "Matches existing sentiment scoring convention"},
        {"Decision": "Fluency score", "Choice": "Qwen perplexity", "Reason": "Lower is better; inverted before harmonic mean"},
        {"Decision": "Baseline", "Choice": "Instruction rewrite with masked diffusion continuation", "Reason": "Direct prompt alternative to activation steering"},
    ]
    write_table_bundle(out_dir, "table_1_experimental_setup", rows, "Cats/dogs experimental setup and design decisions.", "tab:cats-dogs-setup")


def best_rows(cats_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, stage_dir, _, _ in STAGE_SPECS:
        data = read_csv(stage_trial_path(cats_dir, stage_dir))
        best = max(data, key=lambda r: f(r, "harmonic_score"))
        rows.append(compact_steering_row(label, best))
    all_rows = []
    for _, stage_dir, _, _ in STAGE_SPECS:
        all_rows.extend(read_csv(stage_trial_path(cats_dir, stage_dir)))
    overall = max(all_rows, key=lambda r: f(r, "harmonic_score"))
    return rows, overall


def best_config_table(cats_dir: Path, out_dir: Path) -> None:
    rows, overall = best_rows(cats_dir)
    rows.append(compact_steering_row("Overall best steering", overall))
    write_table_bundle(out_dir, "table_2_best_steering_configs", rows, "Best cats/dogs steering configurations from the greedy ablations.", "tab:best-steering")


def ablation_figure(cats_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, (_, stage_dir, key, title) in zip(axes, STAGE_SPECS):
        rows = group_best(read_csv(stage_trial_path(cats_dir, stage_dir)), key)
        labels = [str(r[key]) for r in rows]
        x = np.arange(len(rows))
        vals = [f(r, "harmonic_score") for r in rows]
        bars = ax.bar(x, vals, color=COLORS["harmonic"], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ablated value")
        style_ax(ax)
        best_i = int(np.argmax(vals))
        bars[best_i].set_color("#004D40")
        for i, r in enumerate(rows):
            ax.text(
                i,
                vals[i] + 0.001,
                f"P={fmt(r['target_prob'])}\nPPL={fmt(r['perplexity'], 1)}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
            )
    axes[0].set_ylabel("Harmonic score ↑")
    fig.suptitle("Greedy steering ablations: each panel selects the best setting for that variable", fontsize=12)
    save(fig, out_dir / "figure_1_greedy_ablation_summary.pdf")


def steering_depth_figure(cats_dir: Path, out_dir: Path) -> None:
    rows = read_csv(cats_dir / "asap_refill/asap_hyperparam_trials.csv")
    rows = [
        r
        for r in rows
        if r["refill_steps"] == "5" and r["sampling_temp"] == "0.5" and r["identify_temp"] == "0.5"
    ]
    rows = sorted(rows, key=lambda r: f(r, "k"))
    xs = [f(r, "k") for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(xs, [f(r, "harmonic_score") for r in rows], marker="o", color=COLORS["harmonic"], label="Harmonic ↑")
    ax1.plot(xs, [f(r, "target_prob") for r in rows], marker="s", color=COLORS["target"], label="Target prob ↑")
    ax1.set_xlabel("Steering depth k")
    ax1.set_ylabel("Score ↑")
    ax1.set_xticks(xs)
    style_ax(ax1)
    ax2 = ax1.twinx()
    ax2.plot(xs, [f(r, "perplexity") for r in rows], marker="^", color=COLORS["ppl"], label="PPL ↓")
    ax2.set_ylabel("Perplexity ↓")
    ax2.spines["top"].set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=False)
    ax1.set_title("Repeated steering improves target movement while PPL decreases")
    save(fig, out_dir / "figure_2_steering_depth_k.pdf")


def length_figure(cats_dir: Path, out_dir: Path) -> None:
    _, overall = best_rows(cats_dir)
    rows = read_csv(stage_length_path(cats_dir, STAGE_TO_DIR[overall["stage"]]))
    keys = ["k", "refill_steps", "sampling_temp", "identify_temp"]
    rows = [r for r in rows if all(str(r[k]) == str(overall[k]) for k in keys)]
    rows = sorted(rows, key=lambda r: LENGTH_BIN_ORDER.get(r["length_bin"], 9))
    labels = [r["length_bin"] + f"\n(n={r['num_records']})" for r in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.bar(x - width / 2, [f(r, "target_prob") for r in rows], width, color=COLORS["target"], label="Target prob ↑")
    ax1.bar(x + width / 2, [f(r, "harmonic_score") for r in rows], width, color=COLORS["harmonic"], label="Harmonic ↑")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Score ↑")
    style_ax(ax1)
    ax2 = ax1.twinx()
    ax2.plot(x, [f(r, "perplexity") for r in rows], color=COLORS["ppl"], marker="o", label="PPL ↓")
    ax2.set_ylabel("Perplexity ↓")
    ax2.spines["top"].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)
    ax1.set_title("Sentence length analysis for the best steering setting")
    save(fig, out_dir / "figure_3_sentence_length_analysis.pdf")


def baseline_figure_and_table(cats_dir: Path, out_dir: Path) -> None:
    _, steering = best_rows(cats_dir)
    prompt = next(r for r in read_csv(cats_dir / "prompt_baseline/cats_dogs_prompt_summary.csv") if r["direction"] == "overall")
    rows = [
        {
            "Method": "Activation steering",
            "Config": f"k={steering['k']}, u={steering['refill_steps']}, samp={steering['sampling_temp']}, ident={steering['identify_temp']}",
            "N": steering["num_records"],
            "Target prob ↑": fmt(steering["target_prob"]),
            "PPL ↓": fmt(steering["perplexity"], 2),
            "Harmonic ↑": fmt(steering["harmonic_score"]),
            "Semantic sim ↑": "n/a",
        },
        {
            "Method": "Instruction prompting",
            "Config": "rewrite prompt + masked diffusion",
            "N": prompt["num_records"],
            "Target prob ↑": fmt(prompt["target_prob"]),
            "PPL ↓": fmt(prompt["perplexity"], 2),
            "Harmonic ↑": fmt(prompt["harmonic_score"]),
            "Semantic sim ↑": fmt(prompt["semantic_similarity"]),
        },
    ]
    write_table_bundle(out_dir, "table_3_steering_vs_prompting", rows, "Apples-to-apples cats/dogs comparison on the same 2100 examples.", "tab:steering-vs-prompting")

    labels = [r["Method"] for r in rows]
    target = [float(rows[0]["Target prob ↑"]), float(rows[1]["Target prob ↑"])]
    harmonic = [float(rows[0]["Harmonic ↑"]), float(rows[1]["Harmonic ↑"])]
    ppl = [float(rows[0]["PPL ↓"]), float(rows[1]["PPL ↓"])]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    vals = [(target, "Target token prob ↑", COLORS["target"]), (ppl, "Perplexity ↓", COLORS["ppl"]), (harmonic, "Harmonic score ↑", COLORS["harmonic"])]
    for ax, (ys, title, color) in zip(axes, vals):
        ax.bar(x, ys, color=[COLORS["steer"], COLORS["prompt"]])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(title, fontsize=10.5)
        style_ax(ax)
        for i, y in enumerate(ys):
            ax.text(i, y + max(ys) * 0.02, fmt(y, 3 if y < 1 else 2), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Prompting outperforms steering on the corrected cats/dogs evaluation set", fontsize=12)
    save(fig, out_dir / "figure_4_steering_vs_prompting.pdf")


def direction_table(cats_dir: Path, out_dir: Path) -> None:
    rows = read_csv(cats_dir / "prompt_baseline/cats_dogs_prompt_summary.csv")
    rows = [r for r in rows if r["direction"] != "overall"]
    out_rows = [
        {
            "Method": "Instruction prompting",
            "Direction": r["direction"].replace("_", "→"),
            "N": r["num_records"],
            "Target prob ↑": fmt(r["target_prob"]),
            "PPL ↓": fmt(r["perplexity"], 2),
            "Harmonic ↑": fmt(r["harmonic_score"]),
            "Semantic sim ↑": fmt(r["semantic_similarity"]),
        }
        for r in rows
    ]
    write_table_bundle(out_dir, "table_4_prompt_direction_breakdown", out_rows, "Direction-level prompting baseline results.", "tab:prompt-directions")


def qualitative_table(cats_dir: Path, out_dir: Path) -> None:
    steer = read_json(cats_dir / "asap_refill/asap_qualitative_examples.json")
    prompt = read_json(cats_dir / "prompt_baseline/cats_dogs_prompt_qualitative_examples.json")
    steer_by_id = {r["id"]: r for r in steer}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for p in prompt:
        s = steer_by_id.get(p["id"])
        if s:
            matched.append((s, p))
        if len(matched) >= 4:
            break
    if len(matched) < 4:
        for i in range(min(4 - len(matched), len(steer), len(prompt))):
            matched.append((steer[i], prompt[i]))

    rows: list[dict[str, Any]] = []
    for s, p in matched[:4]:
        original = p.get("original_text") or s.get("original_text", "")
        rows.append(
            {
                "Direction": p.get("direction", s.get("direction", "")).replace("_", "→"),
                "Original": textwrap.shorten(original, width=90, placeholder="..."),
                "Steered": textwrap.shorten(s.get("steered_text", ""), width=90, placeholder="..."),
                "Prompted": textwrap.shorten(p.get("generated_text", ""), width=90, placeholder="..."),
                "Steer H": fmt(s.get("harmonic_score")),
                "Prompt H": fmt(p.get("harmonic_score")),
            }
        )
    write_table_bundle(out_dir, "table_5_qualitative_examples", rows, "Representative cats/dogs qualitative examples.", "tab:qual-examples")


def imdb_appendix(cats_dir: Path, out_dir: Path) -> None:
    rows = read_csv(cats_dir / "imdb_prompt_baseline/imdb_prompt_summary.csv")
    out_rows = [
        {
            "Direction": r["direction"].replace("_", "→"),
            "N": r["num_records"],
            "Target prob ↑": fmt(r["target_prob"]),
            "PPL ↓": fmt(r["perplexity"], 2),
            "Harmonic ↑": fmt(r["harmonic_score"]),
            "Semantic sim ↑": fmt(r["semantic_similarity"]),
        }
        for r in rows
    ]
    write_table_bundle(out_dir, "appendix_table_imdb_prompt_baseline", out_rows, "IMDB prompting baseline results.", "tab:imdb-prompt")

    labels = [r["Direction"] for r in out_rows]
    x = np.arange(len(out_rows))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x - 0.22, [float(r["Target prob ↑"]) for r in out_rows], 0.22, color=COLORS["target"], label="Target prob ↑")
    ax.bar(x, [float(r["Harmonic ↑"]) for r in out_rows], 0.22, color=COLORS["harmonic"], label="Harmonic ↑")
    ax.bar(x + 0.22, [float(r["Semantic sim ↑"]) for r in out_rows], 0.22, color=COLORS["prompt"], label="Semantic sim ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Appendix: IMDB prompting baseline")
    style_ax(ax)
    ax.legend(frameon=False)
    save(fig, out_dir / "appendix_figure_imdb_prompt_baseline.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats-dir", type=Path, default=Path("cats_dogs"))
    ap.add_argument("--output-dir", type=Path, default=Path("cats_dogs/paper_assets"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_table(args.cats_dir, args.output_dir)
    best_config_table(args.cats_dir, args.output_dir)
    ablation_figure(args.cats_dir, args.output_dir)
    steering_depth_figure(args.cats_dir, args.output_dir)
    length_figure(args.cats_dir, args.output_dir)
    baseline_figure_and_table(args.cats_dir, args.output_dir)
    direction_table(args.cats_dir, args.output_dir)
    qualitative_table(args.cats_dir, args.output_dir)
    imdb_appendix(args.cats_dir, args.output_dir)
    print(f"Wrote paper assets to {args.output_dir}")


if __name__ == "__main__":
    main()
