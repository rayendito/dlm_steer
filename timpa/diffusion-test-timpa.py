#!/usr/bin/env python3
"""
Evaluate TIMPA evolution outputs from evolution_all.txt.

Input format example:
prompt 1
original: ...
evol 0  : ...
evol 1  : ...
...
--------------------------------------------------

Outputs sentiment and perplexity scores using eval_dito functions.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from pathlib import Path
from typing import Any

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_dito import score_labels, perplexity


_PROMPT_RE = re.compile(r"^prompt\s+(\d+)\s*$", re.IGNORECASE)
_ORIGINAL_RE = re.compile(r"^original\s*:\s*(.*)$", re.IGNORECASE)
_EVOL_RE = re.compile(r"^evol\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def parse_evolution_file(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    prompts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-"):
            continue

        m_prompt = _PROMPT_RE.match(line)
        if m_prompt:
            if current is not None:
                prompts.append(current)
            current = {
                "prompt_idx": int(m_prompt.group(1)),
                "original": None,
                "evolutions": [],
            }
            continue

        if current is None:
            continue

        m_original = _ORIGINAL_RE.match(line)
        if m_original:
            current["original"] = m_original.group(1)
            continue

        m_evol = _EVOL_RE.match(line)
        if m_evol:
            current["evolutions"].append(
                {
                    "evol_step": int(m_evol.group(1)),
                    "text": m_evol.group(2),
                }
            )
            continue

    if current is not None:
        prompts.append(current)
    return prompts


def build_rows(parsed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in parsed:
        prompt_idx = int(item["prompt_idx"])
        original = item.get("original")
        if isinstance(original, str) and original.strip():
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "kind": "original",
                    "evol_step": -1,
                    "text": original,
                }
            )
        for ev in item.get("evolutions", []):
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "kind": "evol",
                    "evol_step": int(ev["evol_step"]),
                    "text": str(ev["text"]),
                }
            )
    return rows


def compute_averages(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    evol_rows = [r for r in scored_rows if r["kind"] == "evol"]
    original_rows = [r for r in scored_rows if r["kind"] == "original"]

    def _avg(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {
                "avg_positive_sentiment": 0.0,
                "avg_negative_sentiment": 0.0,
                "avg_perplexity": 0.0,
            }
        n = float(len(rows))
        return {
            "avg_positive_sentiment": sum(r["positive_sentiment"] for r in rows) / n,
            "avg_negative_sentiment": sum(r["negative_sentiment"] for r in rows) / n,
            "avg_perplexity": sum(r["perplexity"] for r in rows) / n,
        }

    by_step: dict[int, list[dict[str, Any]]] = {}
    for r in evol_rows:
        step = int(r["evol_step"])
        by_step.setdefault(step, []).append(r)

    per_step = []
    for step in sorted(by_step.keys()):
        entry = {"evol_step": step, "count": len(by_step[step])}
        entry.update(_avg(by_step[step]))
        per_step.append(entry)

    return {
        "overall_all_rows": {
            "count": len(scored_rows),
            **_avg(scored_rows),
        },
        "overall_original_only": {
            "count": len(original_rows),
            **_avg(original_rows),
        },
        "overall_evol_only": {
            "count": len(evol_rows),
            **_avg(evol_rows),
        },
        "evol_avg_by_step": per_step,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate evolution_all.txt with sentiment + perplexity")
    ap.add_argument(
        "--input-file",
        type=Path,
        default=Path("results/test_timpa/timpa_all.txt"),
        help="Path to evolution_all.txt",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/test_timpa/test_timpa_eval.json"),
        help="Output JSON path",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Mini-batch size for scoring")
    args = ap.parse_args()

    if not args.input_file.is_file():
        print(f"Missing input file: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    if args.batch_size <= 0:
        print("--batch-size must be > 0", file=sys.stderr)
        sys.exit(1)

    parsed = parse_evolution_file(args.input_file)
    rows = build_rows(parsed)
    if not rows:
        print("No rows found in evolution file.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    scored_rows: list[dict[str, Any]] = []
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = rows[start:end]
        texts = [r["text"] for r in batch]
        try:
            _, p_pos, p_neg = score_labels(texts)
            ppls = perplexity(texts)
        except Exception as e:
            print(f"Scoring failed in batch [{start}:{end}]: {e}", file=sys.stderr)
            raise

        for i, r in enumerate(batch):
            out = dict(r)
            out["positive_sentiment"] = float(p_pos[i].item())
            out["negative_sentiment"] = float(p_neg[i].item())
            out["perplexity"] = float(ppls[i].item())
            scored_rows.append(out)

        print(f"Scored batch {start}-{end - 1} / {total - 1}", file=sys.stderr)

    # Per-prompt grouped output for easier downstream analysis
    grouped: dict[int, dict[str, Any]] = {}
    for r in scored_rows:
        pi = int(r["prompt_idx"])
        g = grouped.setdefault(pi, {"prompt_idx": pi, "original": None, "evolutions": []})
        if r["kind"] == "original":
            g["original"] = r
        else:
            g["evolutions"].append(r)
    for g in grouped.values():
        g["evolutions"] = sorted(g["evolutions"], key=lambda x: x["evol_step"])
    averages = compute_averages(scored_rows)

    payload = {
        "input_file": str(args.input_file),
        "scorer": "eval_dito.score_labels + eval_dito.perplexity",
        "num_rows": len(scored_rows),
        "averages": averages,
        "rows": scored_rows,
        "by_prompt": [grouped[k] for k in sorted(grouped.keys())],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out_json}", file=sys.stderr)
    print(
        (
            "Final evol avg -> "
            f"pos={averages['overall_evol_only']['avg_positive_sentiment']:.4f}, "
            f"neg={averages['overall_evol_only']['avg_negative_sentiment']:.4f}, "
            f"ppl={averages['overall_evol_only']['avg_perplexity']:.4f}"
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

