#!/usr/bin/env python3
"""Export saved TIMPA-steer masking spans as token-level HTML visualizations.

The renderer mirrors the ``timpa_experimental.visualize_timpa_steer_results``
markup used for TIMPA-probabilistic result pages, but reads completed trial
artifacts rather than loading a diffusion model again.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE_DIRECTORY_NAMES = {
    "masking-policy-selected": "selected",
    "masking-policy-random": "random",
    "masking-policy-aggressive-misalignment": "aggressive_misalignment",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render completed TIMPA-steer masking spans into HTML pages."
    )
    parser.add_argument(
        "--merged",
        type=Path,
        nargs="+",
        required=True,
        help="Merged TIMPA-steer result JSON files containing control trials.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="The repository main_results directory.",
    )
    return parser.parse_args()


def _record_key(record: dict[str, Any]) -> tuple[int, str]:
    identifier = str(record.get("source_id") or record["example_id"])
    match = re.search(r"-(\d+)(?:-[^-]+)?$", identifier)
    if match is None:
        raise ValueError(f"Could not determine example order from {identifier!r}.")
    return int(match.group(1)), identifier


def _safe_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(record["text"])
    spans = record.get("mask_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"Record {record['example_id']!r} has no masking spans.")
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError(
                f"Record {record['example_id']!r} has invalid masking offsets."
            )
    return spans


def _highlighted_text(text: str, spans: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        probability = float(span["probability"])
        strength = min(max(probability, 0.0), 1.0)
        tooltip = f"masking probability: {probability:.6g}"
        pieces.append(
            '<span class="token" '
            f'style="background-color: rgba(220, 38, 38, {strength:.3f})" '
            f'data-tooltip="{html.escape(tooltip, quote=True)}" '
            f'title="{html.escape(tooltip, quote=True)}">'
            f"{html.escape(text[start:end])}</span>"
        )
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _masked_text(text: str, spans: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        start, end = int(span["start"]), int(span["end"])
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        fragment = text[start:end]
        if bool(span["masked"]):
            leading_space_count = len(fragment) - len(fragment.lstrip())
            pieces.append(html.escape(fragment[:leading_space_count]))
            pieces.append(
                '<span class="sampled-mask" '
                f'title="masked token: {html.escape(fragment, quote=True)}">'
                "&lt;|mdm_mask|&gt;</span>"
            )
        else:
            pieces.append(html.escape(fragment))
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _mask_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    masked = total = 0
    for record in records:
        spans = _safe_spans(record)
        masked += sum(bool(span["masked"]) for span in spans)
        total += len(spans)
    if total == 0 or masked > total:
        raise ValueError("Invalid TIMPA-steer masking counts.")
    return masked, total


def _meta(config: dict[str, Any], masked: int, total: int) -> str:
    layers = ", ".join(str(layer) for layer in config.get("steer_layers", ()))
    detection = str(config.get("detection_strategy", "model"))
    detection_details = (
        f"<b>Random mask probability:</b> {float(config['random_mask_probability']):g}"
        if detection == "random"
        else (
            f"<b>Temperature:</b> {float(config['temperature']):g} · "
            f"<b>Margin:</b> {float(config['margin']):g}"
        )
    )
    return (
        "<b>Diffusion model:</b> GSAI-ML/LLaDA-8B-Instruct · "
        f"<b>Steering layers:</b> {html.escape(layers)} · "
        f"<b>Alpha:</b> {float(config['alpha']):g} · "
        f"<b>Detection:</b> {html.escape(detection)} · "
        f"{detection_details} · "
        f"<b>Refill steps:</b> {int(config['refill_steps'])} · "
        f"<b>Diffusion-token mask rate:</b> {masked}/{total} ({masked / total:.1%})"
    )


def _document(
    *,
    dataset: str,
    target: str,
    seed: int,
    config: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    masked, total = _mask_counts(records)
    cards = []
    for index, record in enumerate(records, start=1):
        text = str(record["text"])
        spans = _safe_spans(record)
        cards.append(
            '<section class="card">'
            f'<div class="label">Example {index}</div>'
            '<div class="label">System prompt</div>'
            '<div class="prompt">You are a helpful assistant</div>'
            '<div class="label">Masking probability</div>'
            f'<div class="text">{_highlighted_text(text, spans)}</div>'
            '<div class="label">Masked text</div>'
            f'<div class="text">{_masked_text(text, spans)}</div>'
            '<div class="label">Steered text</div>'
            f'<div class="text">{html.escape(str(record["after"]))}</div>'
            '</section>'
        )
    title = f"TIMPA-steer {dataset} · seed {seed} · {target}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ max-width: 960px; margin: 40px auto; padding: 0 20px; color: #242424;
       background: #fafafa; font: 16px/1.6 system-ui, sans-serif; }}
h1 {{ margin-bottom: 4px; }}
.meta, .legend {{ color: #666; margin-bottom: 20px; }}
.high-probability {{ color: rgb(220, 38, 38); }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px;
         margin: 18px 0; padding: 20px; }}
.label {{ color: #777; font-size: 12px; font-weight: 700; letter-spacing: .08em;
          margin-top: 10px; text-transform: uppercase; }}
.prompt, .text {{ white-space: pre-wrap; }}
.sampled-mask {{ background: #242424; border-radius: 3px; color: white;
                 padding: 1px 3px; }}
.token {{ border-radius: 3px; cursor: help; position: relative; }}
.token:hover::after {{
  background: #242424; border-radius: 5px; bottom: calc(100% + 7px); color: white;
  content: attr(data-tooltip); font-size: 12px; left: 50%; padding: 4px 7px;
  pointer-events: none; position: absolute; transform: translateX(-50%);
  white-space: nowrap; z-index: 10;
}}
</style>
</head>
<body>
<h1>TIMPA steering token identification</h1>
<div class="meta">{_meta(config, masked, total)}</div>
<div class="legend">More intense <span class="high-probability">red</span>
means higher masking probability.</div>
{''.join(cards)}
</body>
</html>
"""


def _load_groups(paths: list[Path]) -> dict[tuple[str, int, str, str], tuple[dict[str, Any], list[dict[str, Any]]]]:
    groups: dict[tuple[str, int, str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    seen_examples: dict[tuple[str, int, str, str], set[str]] = defaultdict(set)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            trials = json.load(handle).get("trials", [])
        if not trials:
            raise ValueError(f"{path} contains no trials.")
        for trial in trials:
            stage = str(trial["stage"])
            if stage not in STAGE_DIRECTORY_NAMES:
                continue
            config = trial["config"]
            dataset, seed = str(config["dataset"]), int(config["seed"])
            for record in trial.get("records", []):
                target = str(record["target_direction"])
                key = (dataset, seed, stage, target)
                if key not in groups:
                    groups[key] = (config, [])
                saved_config, records = groups[key]
                if saved_config != config:
                    raise ValueError(f"Inconsistent configurations in {key!r}.")
                example_id = str(record["example_id"])
                if example_id in seen_examples[key]:
                    raise ValueError(f"Duplicate record {example_id!r} in {key!r}.")
                seen_examples[key].add(example_id)
                records.append(record)
    if not groups:
        raise ValueError("No supported TIMPA-steer control trials were found.")
    return groups


def export_html(merged_paths: list[Path], output_root: Path) -> list[Path]:
    """Write one token-level masking visualization per dataset/seed/target."""
    groups = _load_groups(merged_paths)
    written = []
    for (dataset, seed, stage, target), (config, records) in sorted(groups.items()):
        ordered = sorted(records, key=_record_key)
        if len(ordered) != 100:
            raise ValueError(f"Expected 100 records for {(dataset, seed, stage, target)!r}.")
        condition = STAGE_DIRECTORY_NAMES[stage]
        output_path = (
            Path(output_root)
            / f"timpa_steer_{dataset}_test_html"
            / f"seed{seed}"
            / condition
            / f"{target}.html"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _document(
                dataset=dataset,
                target=target,
                seed=seed,
                config=config,
                records=ordered,
            ),
            encoding="utf-8",
        )
        written.append(output_path)
    return written


def main() -> int:
    args = _parse_args()
    for path in export_html(args.merged, args.output_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
