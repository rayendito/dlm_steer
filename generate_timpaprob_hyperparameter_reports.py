#!/usr/bin/env python3
"""Generate IMDb and Cat/Dog TIMPA-probabilistic sweep summaries."""

from __future__ import annotations

import base64
import csv
import html
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup


RESULTS_ROOT = Path("timpateks_results")
TEMPERATURES = (0.25, 0.5, 1.0, 2.0)
MARGINS = (0.001, 0.1, 0.25, 0.5)
EXPECTED_EXAMPLES_PER_DIRECTION = 10
EVALUATION_FILENAME = "evaluation.csv"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    directions: tuple[str, str]
    recommendation: str
    output: Path


SPECS = (
    DatasetSpec(
        slug="imdb",
        title="IMDb TIMPA-probabilistic hyperparameter search",
        directions=("positive", "negative"),
        recommendation="temp0.25_margin0.1",
        output=Path("imdb_timpaprob_hyperparameter_search_summary.html"),
    ),
    DatasetSpec(
        slug="catdog",
        title="Cat/Dog TIMPA-probabilistic hyperparameter search",
        directions=("cat", "dog"),
        recommendation="temp1_margin0.5",
        output=Path("catdog_timpaprob_hyperparameter_search_summary.html"),
    ),
)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def percent(value, digits=0):
    return f"{100 * value:.{digits}f}%"


def signed_probability(value):
    return f"{value:+.3f}"


def _read_evaluation(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sentence_id",
            "succesfully_steered",
            "target_probability_gain",
            "faithfulness_score",
            "structure_retention_score",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != 20:
        raise ValueError(f"Expected 20 rows in {path}, found {len(rows)}.")
    return rows


def _mask_counts(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    cards = soup.select("section.card")
    if len(cards) != EXPECTED_EXAMPLES_PER_DIRECTION:
        raise ValueError(f"Expected 10 visualization cards in {path}.")
    masked = sum(len(card.select("span.sampled-mask")) for card in cards)
    total = sum(len(card.select("span.token")) for card in cards)
    if total == 0 or masked > total:
        raise ValueError(f"Invalid mask counts in {path}: {masked}/{total}.")
    return masked, total


def _configuration_metadata(description):
    if description in {"negative_delta", "random"}:
        return None, None
    temperature_text, margin_text = description.removeprefix("temp").split(
        "_margin", maxsplit=1
    )
    return float(temperature_text), float(margin_text)


def load_results(spec):
    csv_root = RESULTS_ROOT / f"{spec.slug}_sweep_csv" / "seed42"
    html_root = RESULTS_ROOT / f"{spec.slug}_sweep_html" / "seed42"
    paths = sorted(csv_root.glob(f"*/{EVALUATION_FILENAME}"))
    if len(paths) != 18:
        raise RuntimeError(
            f"Expected 18 evaluation files for {spec.slug}, found {len(paths)}."
        )

    results = []
    for path in paths:
        description = path.parent.name
        temperature, margin = _configuration_metadata(description)
        rows = _read_evaluation(path)
        direction_rows = {
            spec.directions[0]: rows[:EXPECTED_EXAMPLES_PER_DIRECTION],
            spec.directions[1]: rows[EXPECTED_EXAMPLES_PER_DIRECTION:],
        }
        result = {
            "configuration": description,
            "temperature": temperature,
            "margin": margin,
            "baseline": description in {"negative_delta", "random"},
        }

        all_masked = 0
        all_tokens = 0
        for direction in spec.directions:
            selected = direction_rows[direction]
            successes = [int(row["succesfully_steered"]) for row in selected]
            probability_gains = [
                float(row["target_probability_gain"]) for row in selected
            ]
            faithfulness = [int(row["faithfulness_score"]) for row in selected]
            structure = [
                int(row["structure_retention_score"]) for row in selected
            ]
            masked, total = _mask_counts(
                html_root / description / f"{direction}.html"
            )
            all_masked += masked
            all_tokens += total
            result[f"{direction}_success"] = mean(successes)
            result[f"{direction}_probability_gain"] = mean(probability_gains)
            result[f"{direction}_faithfulness"] = mean(faithfulness)
            result[f"{direction}_structure"] = mean(structure)
            result[f"{direction}_mask_rate"] = masked / total

        successes = [int(row["succesfully_steered"]) for row in rows]
        probability_gains = [
            float(row["target_probability_gain"]) for row in rows
        ]
        faithfulness = [int(row["faithfulness_score"]) for row in rows]
        structure = [int(row["structure_retention_score"]) for row in rows]
        result.update(
            success=mean(successes),
            probability_gain=mean(probability_gains),
            faithfulness=mean(faithfulness),
            structure=mean(structure),
            quality=mean(faithfulness + structure),
            mask_rate=all_masked / all_tokens,
            direction_gap=abs(
                result[f"{spec.directions[0]}_success"]
                - result[f"{spec.directions[1]}_success"]
            ),
        )
        results.append(result)
    return results


def pareto_frontier(results):
    frontier = set()
    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            no_worse = (
                other["success"] >= candidate["success"]
                and other["probability_gain"] >= candidate["probability_gain"]
                and other["quality"] >= candidate["quality"]
                and other["mask_rate"] <= candidate["mask_rate"]
            )
            strictly_better = (
                other["success"] > candidate["success"]
                or other["probability_gain"] > candidate["probability_gain"]
                or other["quality"] > candidate["quality"]
                or other["mask_rate"] < candidate["mask_rate"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(candidate["configuration"])
    return frontier


def rank_results(results):
    return sorted(
        results,
        key=lambda row: (
            -row["success"],
            -row["probability_gain"],
            -row["quality"],
            -row["faithfulness"],
            row["mask_rate"],
            row["configuration"],
        ),
    )


def _plot_data(results, metric):
    sampled = {
        (row["temperature"], row["margin"]): row[metric]
        for row in results
        if not row["baseline"]
    }
    return np.array(
        [
            [sampled[(temperature, margin)] for margin in MARGINS]
            for temperature in TEMPERATURES
        ]
    )


def tradeoff_figure(results):
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    plots = (
        ("success", "Steering success", 0, 1, "YlGnBu", lambda value: f"{value:.0%}"),
        ("probability_gain", "Target probability gain", -1, 1, "RdYlGn", lambda value: f"{value:+.2f}"),
        ("faithfulness", "Faithfulness", 1, 5, "YlGnBu", lambda value: f"{value:.2f}"),
        ("structure", "Structure retention", 1, 5, "YlGnBu", lambda value: f"{value:.2f}"),
        ("quality", "Mean judge quality", 1, 5, "YlGnBu", lambda value: f"{value:.2f}"),
        ("mask_rate", "Diffusion-token mask rate", 0, 0.6, "YlGnBu", lambda value: f"{value:.0%}"),
    )
    for axis, (metric, title, vmin, vmax, cmap, formatter) in zip(axes.flat, plots):
        values = _plot_data(results, metric)
        image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xticks(range(len(MARGINS)), [f"{value:g}" for value in MARGINS])
        axis.set_yticks(
            range(len(TEMPERATURES)), [f"{value:g}" for value in TEMPERATURES]
        )
        axis.set_xlabel("Margin")
        axis.set_ylabel("Temperature")
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                normalized = (value - vmin) / (vmax - vmin)
                axis.text(
                    column_index,
                    row_index,
                    formatter(value),
                    ha="center",
                    va="center",
                    color="white" if normalized > 0.68 else "#202529",
                    fontsize=8.5,
                    weight="semibold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def badge(text, css_class):
    return f'<span class="badge {css_class}">{html.escape(text)}</span>'


def build_report(spec, results):
    by_name = {row["configuration"]: row for row in results}
    recommended = by_name[spec.recommendation]
    ranked = rank_results(results)
    frontier = pareto_frontier(results)
    negative_delta = by_name["negative_delta"]
    random = by_name["random"]
    best_observed = ranked[0]
    direction_a, direction_b = spec.directions

    if spec.slug == "catdog":
        decision = (
            f"<strong>Select <code>{recommended['configuration']}</code> "
            f"provisionally.</strong> It ties for the highest steering success "
            f"({percent(recommended['success'])}) with a "
            f"{signed_probability(recommended['probability_gain'])} mean target "
            f"probability gain, and has the best combined judge "
            f"quality among the least-invasive members of that top tier: "
            f"{recommended['faithfulness']:.2f} faithfulness, "
            f"{recommended['structure']:.2f} structure retention, and an "
            f"{percent(recommended['mask_rate'], 1)} mask rate."
        )
        findings = (
            f"Twelve sampled settings tie at {percent(recommended['success'])}; "
            "the recommendation therefore depends on quality and masking, not "
            "classification alone.",
            f"At the recommendation, {direction_a} steering succeeds on "
            f"{percent(recommended[f'{direction_a}_success'])} and {direction_b} "
            f"steering on {percent(recommended[f'{direction_b}_success'])}.",
            f"<code>temp1_margin0.001</code> has the same steering success, "
            "target probability gain, and quality mean, but masks more tokens "
            "(26.1% versus 18.3%).",
            f"The deterministic <code>negative_delta</code> baseline masks "
            f"{percent(negative_delta['mask_rate'], 1)} of tokens and reaches "
            f"only {percent(negative_delta['success'])} success.",
            f"Random masking falls to {percent(random['success'])}, confirming "
            "that the AR-DLM token scores add substantial targeting value.",
        )
        caveat = (
            "The recommendation is selected from a broad 90%-success tie and "
            "faithfulness remains low at 2.25/5. Repeat the leading settings over "
            "several seeds and more examples before fixing the final parameters."
        )
        headline = "1.0 / 0.5"
    else:
        decision = (
            f"<strong>No IMDb configuration demonstrates reliable steering.</strong> "
            f"Use <code>{recommended['configuration']}</code> only as the least-bad "
            f"sampled fallback: it reaches {percent(recommended['success'])} "
            f"success with a {signed_probability(recommended['probability_gain'])} "
            f"mean target probability gain, "
            f"{recommended['faithfulness']:.2f} faithfulness, "
            f"{recommended['structure']:.2f} structure retention, and an "
            f"{percent(recommended['mask_rate'], 1)} mask rate. The nominal winner, "
            f"<code>negative_delta</code>, reaches just "
            f"{percent(negative_delta['success'])} while sharply degrading quality."
        )
        findings = (
            f"The aggressive <code>negative_delta</code> baseline is highest on "
            f"steering success ({percent(negative_delta['success'])}), but masks "
            f"{percent(negative_delta['mask_rate'], 1)} of tokens and scores only "
            f"{negative_delta['quality']:.2f}/5 in mean judge quality.",
            f"The sampled fallback succeeds on only "
            f"{percent(recommended[f'{direction_a}_success'])} of {direction_a} "
            f"targets and {percent(recommended[f'{direction_b}_success'])} of "
            f"{direction_b} targets.",
            f"<code>temp0.5_margin0.001</code> ties the fallback on steering and "
            "quality mean, but has a lower target probability gain (+0.138 versus "
            "+0.148), slightly higher faithfulness, and lower structure retention.",
            "Higher temperature and margin preserve more content, but IMDb "
            "steering success approaches zero as masking becomes conservative.",
            f"Random masking reaches only {percent(random['success'])} and has "
            f"{random['quality']:.2f}/5 mean judge quality.",
        )
        caveat = (
            "Do not lock in these IMDb hyperparameters. The best sampled setting "
            "converts only 4 of 20 examples, and the aggressive baseline converts "
            "6 of 20. The detection prompts, task formulation, or refill behavior "
            "should be revised before a larger confirmation sweep."
        )
        headline = "0.25 / 0.1"

    key_names = [
        recommended["configuration"],
        best_observed["configuration"],
        "negative_delta",
        "random",
    ]
    key_rows = []
    seen = set()
    for name in key_names:
        if name in seen:
            continue
        seen.add(name)
        row = by_name[name]
        labels = []
        if name == recommended["configuration"]:
            labels.append(badge("recommended", "recommended"))
        if row["baseline"]:
            labels.append(badge("baseline", "baseline"))
        key_rows.append(
            "<tr>"
            f"<td><code>{html.escape(name)}</code> {' '.join(labels)}</td>"
            f"<td class=\"num\">{percent(row['success'])}</td>"
            f"<td class=\"num\">{signed_probability(row['probability_gain'])}</td>"
            f"<td class=\"num\">{row['faithfulness']:.2f}</td>"
            f"<td class=\"num\">{row['structure']:.2f}</td>"
            f"<td class=\"num\">{row['quality']:.2f}</td>"
            f"<td class=\"num\">{percent(row['mask_rate'], 1)}</td>"
            "</tr>"
        )

    direction_rows = []
    for direction in spec.directions:
        direction_rows.append(
            "<tr>"
            f"<td>{html.escape(direction.title())}</td>"
            f"<td class=\"num\">{percent(recommended[f'{direction}_success'])}</td>"
            f"<td class=\"num\">{signed_probability(recommended[f'{direction}_probability_gain'])}</td>"
            f"<td class=\"num\">{recommended[f'{direction}_faithfulness']:.2f}</td>"
            f"<td class=\"num\">{recommended[f'{direction}_structure']:.2f}</td>"
            f"<td class=\"num\">{percent(recommended[f'{direction}_mask_rate'], 1)}</td>"
            "</tr>"
        )

    all_rows = []
    for rank, row in enumerate(ranked, start=1):
        labels = []
        if row["configuration"] == recommended["configuration"]:
            labels.append(badge("recommended", "recommended"))
        if row["baseline"]:
            labels.append(badge("baseline", "baseline"))
        elif row["configuration"] in frontier:
            labels.append(badge("Pareto", "pareto"))
        all_rows.append(
            "<tr>"
            f"<td class=\"num\">{rank}</td>"
            f"<td><code>{html.escape(row['configuration'])}</code> {' '.join(labels)}</td>"
            f"<td class=\"num\">{percent(row['success'])}</td>"
            f"<td class=\"num\">{signed_probability(row['probability_gain'])}</td>"
            f"<td class=\"num\">{percent(row[f'{direction_a}_success'])}</td>"
            f"<td class=\"num\">{percent(row[f'{direction_b}_success'])}</td>"
            f"<td class=\"num\">{row['faithfulness']:.2f}</td>"
            f"<td class=\"num\">{row['structure']:.2f}</td>"
            f"<td class=\"num\">{row['quality']:.2f}</td>"
            f"<td class=\"num\">{percent(row['mask_rate'], 1)}</td>"
            "</tr>"
        )

    figure = tradeoff_figure(results)
    findings_html = "".join(f"<li>{item}</li>" for item in findings)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(spec.title)} executive summary</title>
<style>
:root{{--ink:#202529;--muted:#626b71;--line:#d8dee1;--paper:#fff;--wash:#f3f5f6;--teal:#287a70;--orange:#c66b2b;--blue:#3d6f9d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.58 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{background:#26333b;color:#fff;padding:38px 24px 32px}}header .inner,main{{max-width:1160px;margin:auto}}h1{{margin:0 0 7px;font-size:30px}}header p{{margin:0;color:#dce3e6}}
main{{background:var(--paper);padding:30px 36px 58px}}h2{{font-size:21px;margin:38px 0 11px}}h3{{font-size:16px;margin:0 0 8px}}p{{margin:8px 0 14px}}
.stats{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid var(--line)}}.stat{{padding:16px;border-right:1px solid var(--line)}}.stat:last-child{{border:0}}.stat b{{display:block;font-size:23px;font-variant-numeric:tabular-nums}}.stat span{{display:block;color:var(--muted);font-size:12px}}
.callout{{border-left:4px solid var(--teal);background:#eef7f5;padding:15px 18px;margin:18px 0}}.callout strong{{color:#17564f}}.warning{{border-left-color:var(--orange);background:#fff7ea}}.warning strong{{color:#754516}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}}.panel{{border:1px solid var(--line);border-radius:6px;padding:17px}}.panel p:last-child{{margin-bottom:0}}
figure{{margin:14px 0 25px}}figure img{{display:block;width:100%;height:auto;border:1px solid var(--line)}}figcaption{{margin-top:7px;color:var(--muted);font-size:13px}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th{{background:#eef1f2;text-align:left}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}tbody tr:last-child td{{border-bottom:0}}.num{{text-align:right;white-space:nowrap}}
code{{background:#edf1f2;border-radius:3px;padding:1px 4px;font-size:.92em}}.badge{{display:inline-block;margin-left:5px;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700;text-transform:uppercase;vertical-align:1px}}.recommended{{background:#d7eee9;color:#17564f}}.baseline{{background:#fae5d5;color:#7b3e12}}.pareto{{background:#dce8f2;color:#28557b}}
.findings{{padding-left:21px}}.findings li{{margin:8px 0}}.table-wrap{{overflow-x:auto}}.footnote{{font-size:13px;color:var(--muted)}}
@media(max-width:780px){{main{{padding:22px 15px 42px}}.stats{{grid-template-columns:1fr 1fr}}.stat:nth-child(2){{border-right:0}}.stat:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.grid{{grid-template-columns:1fr}}h1{{font-size:26px}}}}
</style>
</head>
<body>
<header><div class="inner"><h1>{html.escape(spec.title)}</h1><p>Executive summary of 18 TIMPA configurations on 20 generated texts per configuration</p></div></header>
<main>
<section class="stats">
<div class="stat"><b>{headline}</b><span>recommended temperature / margin</span></div>
<div class="stat"><b>{percent(recommended['success'])}</b><span>steering success</span></div>
<div class="stat"><b>{signed_probability(recommended['probability_gain'])}</b><span>mean target probability gain</span></div>
<div class="stat"><b>{recommended['quality']:.2f}</b><span>mean judge quality on a 1–5 scale</span></div>
<div class="stat"><b>{percent(recommended['mask_rate'], 1)}</b><span>diffusion-token mask rate</span></div>
</section>

<h2>Executive decision</h2>
<div class="callout">{decision}</div>
<ul class="findings">{findings_html}</ul>

<h2>Key configurations</h2>
<div class="table-wrap"><table>
<thead><tr><th>Configuration</th><th class="num">Steering</th><th class="num">Target prob. gain</th><th class="num">Faithfulness</th><th class="num">Structure</th><th class="num">Quality mean</th><th class="num">Mask rate</th></tr></thead>
<tbody>{''.join(key_rows)}</tbody>
</table></div>

<h2>Tradeoff map</h2>
<figure><img src="data:image/png;base64,{figure}" alt="Heatmaps of steering success, target probability gain, faithfulness, structure retention, judge quality, and mask rate across temperature and margin"><figcaption>Sampled configurations only. Conservative masking generally improves preservation, but its effect on steering differs sharply between IMDb and Cat/Dog.</figcaption></figure>

<h2>Recommended setting by target</h2>
<div class="grid">
<section class="panel"><h3>Per-direction results</h3><table><thead><tr><th>Target</th><th class="num">Steering</th><th class="num">Prob. gain</th><th class="num">Faithfulness</th><th class="num">Structure</th><th class="num">Mask rate</th></tr></thead><tbody>{''.join(direction_rows)}</tbody></table></section>
<section class="panel"><h3>Interpretation</h3><p>The same temperature and margin are applied to both target directions. Direction-level rows contain 10 examples each, so every success represents a 10-percentage-point change.</p><p>Probability gain is <code>P(target | after) − P(target | before)</code> under the Qwen forced-choice classifier; positive is better. Faithfulness and structure are judged independently from 1 to 5; higher is better. Mask rate is pooled over diffusion tokens in the corresponding visualization.</p></section>
</div>

<h2>All configurations</h2>
<p class="footnote">Sorted first by higher steering success, then by higher target probability gain, mean judge quality, faithfulness, and lower mask rate. This expresses the steering objective; it is not an additional learned metric.</p>
<div class="table-wrap"><table>
<thead><tr><th class="num">Rank</th><th>Configuration</th><th class="num">Steering</th><th class="num">Prob. gain</th><th class="num">{html.escape(direction_a.title())}</th><th class="num">{html.escape(direction_b.title())}</th><th class="num">Faith.</th><th class="num">Structure</th><th class="num">Quality</th><th class="num">Mask rate</th></tr></thead>
<tbody>{''.join(all_rows)}</tbody>
</table></div>

<h2>Selection caveat</h2>
<div class="callout warning"><strong>This is a provisional validation choice.</strong> {caveat}</div>

<h2>Methodology</h2>
<div class="grid">
<section class="panel"><h3>Generation</h3><p><code>GSAI-ML/LLaDA-8B-Instruct</code> performs 32 confidence-ordered refill steps at sampling temperature 0.1. <code>Qwen/Qwen2.5-32B-Instruct</code> supplies target-versus-source token log-probability deltas. The sweep covers four detection temperatures and four margins, plus deterministic negative-delta and 50% random-mask baselines.</p></section>
<section class="panel"><h3>Evaluation</h3><p><code>Qwen/Qwen2.5-32B-Instruct</code> measures successful target classification and target probability gain, defined as target class probability after minus before rewriting. <code>openai/gpt-4.1-mini</code> scores source faithfulness and discourse-structure retention from 1 to 5. Each configuration contains 10 examples per direction and uses seed 42.</p></section>
</div>
<p class="footnote">Generated from the 18 <code>evaluation.csv</code> files under <code>timpateks_results/{spec.slug}_sweep_csv/seed42/</code>. Mask rates were recovered from the corresponding token-level HTML visualizations.</p>
</main>
</body>
</html>
"""


def main():
    for spec in SPECS:
        report = build_report(spec, load_results(spec))
        spec.output.write_text(report, encoding="utf-8")
        print(f"Wrote {spec.output}")


if __name__ == "__main__":
    main()
