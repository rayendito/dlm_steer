#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llada.configuration_llada import LLaDAConfig
from llada.generate import resteer_v2
from llada.modeling_llada import LLaDAModelLM


MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_STEER_SEQ_LEN = 1024


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def harmonic(a: float, b: float, eps: float = 1e-12) -> float:
    return (2.0 * a * b) / (a + b + eps)


def inv_ppl_scores(ppls: list[float]) -> list[float]:
    vals = np.array(ppls, dtype=float)
    finite = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=float)
    if not finite.any():
        return out.tolist()
    lo = float(np.nanpercentile(vals[finite], 5))
    hi = float(np.nanpercentile(vals[finite], 95))
    clipped = np.clip(vals, lo, hi)
    if hi > lo:
        out = 1.0 - ((clipped - lo) / (hi - lo))
    return out.tolist()


def read_rows(path: Path, max_per_class: int | None, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            concept = (row.get("concept") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if concept in {"cat", "dog"} and text:
                rows.append(
                    {
                        "id": row.get("id") or str(len(rows)),
                        "concept": concept,
                        "text": text,
                        "word_count": len(text.split()),
                    }
                )
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for concept in ("cat", "dog"):
        subset = [r for r in rows if r["concept"] == concept]
        rng.shuffle(subset)
        if max_per_class is not None:
            subset = subset[:max_per_class]
        out.extend(subset)
    rng.shuffle(out)
    assign_length_bins(out)
    return out


def assign_length_bins(rows: list[dict[str, Any]]) -> None:
    lengths = np.array([r["word_count"] for r in rows], dtype=float)
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3])
    for row in rows:
        if row["word_count"] <= q1:
            row["length_bin"] = "short"
        elif row["word_count"] <= q2:
            row["length_bin"] = "medium"
        else:
            row["length_bin"] = "long"


def load_steer_vectors(vectors_path: Path, layer: int, alpha: float, device: str) -> dict[str, torch.Tensor]:
    raw = torch.load(vectors_path, map_location=device)
    pos = raw["positive"]
    neg = raw["negative"]
    cat_to_dog = alpha * (
        l2_normalize(neg[layer].to(device=device, dtype=torch.bfloat16))
        - l2_normalize(pos[layer].to(device=device, dtype=torch.bfloat16))
    )
    dog_to_cat = -cat_to_dog
    return {"cat_to_dog": cat_to_dog, "dog_to_cat": dog_to_cat}


class AnimalScorer:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_ID).eval().to(device)
        self.cat_id = self.tokenizer.encode(" cat", add_special_tokens=False)[0]
        self.dog_id = self.tokenizer.encode(" dog", add_special_tokens=False)[0]

    @torch.no_grad()
    def target_probs(self, texts: list[str], target: str, batch_size: int) -> list[float]:
        tid = self.dog_id if target == "dog" else self.cat_id
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            prompts = [f"Text: {t}\nAnimal:" for t in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            next_logits = logits[torch.arange(len(chunk), device=self.device), last_idx]
            probs = F.softmax(next_logits, dim=-1)
            out.extend(probs[:, tid].detach().cpu().float().tolist())
        return out

    @torch.no_grad()
    def perplexities(self, texts: list[str], batch_size: int) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            raw = [("" if t is None else str(t)) for t in texts[i : i + batch_size]]
            empty_ix = [j for j, t in enumerate(raw) if not t.strip()]
            safe = [t if t.strip() else "." for t in raw]
            enc = self.tokenizer(safe, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**enc).logits
            shift_logits = logits[:, :-1]
            shift_labels = enc["input_ids"][:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(shift_labels.shape)
            mask = enc["attention_mask"][:, 1:]
            denom = mask.sum(dim=1).clamp(min=1)
            ppl = torch.exp((loss * mask).sum(dim=1) / denom).detach().cpu().float().tolist()
            for j in empty_ix:
                ppl[j] = float("inf")
            out.extend(ppl)
        return out


def evaluate_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ppls = [float(r["perplexity"]) for r in records]
    invs = inv_ppl_scores(ppls)
    for row, inv in zip(records, invs):
        row["inv_ppl_norm"] = float(inv)
        row["harmonic_score"] = float(harmonic(float(row["target_prob"]), float(inv)))
    finite_ppl = [p for p in ppls if math.isfinite(p)]
    return (
        {
            "target_prob": float(np.mean([r["target_prob"] for r in records])) if records else 0.0,
            "perplexity": float(np.mean(finite_ppl)) if finite_ppl else float("inf"),
            "harmonic_score": float(np.mean([r["harmonic_score"] for r in records])) if records else 0.0,
            "num_records": len(records),
        },
        records,
    )


def aggregate_by_length(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_bin in ("short", "medium", "long"):
        subset = [r for r in records if r["length_bin"] == length_bin]
        if not subset:
            continue
        agg, _ = evaluate_records([dict(r) for r in subset])
        rows.append({"length_bin": length_bin, **agg})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv", type=Path, default=Path("benchmarks/cats_dogs/train.csv"))
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["refill", "sampling_temp", "identify_temp"], required=True)
    ap.add_argument("--layer", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--resteer-steps", type=int, default=5)
    ap.add_argument("--refill-steps", type=int, nargs="+", required=True)
    ap.add_argument("--sampling-temp", type=float, nargs="+", required=True)
    ap.add_argument("--identify-temp", type=float, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--steer-batch-size", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.dataset_csv, args.max_per_class, args.seed)
    cfg = LLaDAConfig.from_pretrained(MODEL_ID)
    model = LLaDAModelLM.from_pretrained(MODEL_ID, config=cfg, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    base_vectors = load_steer_vectors(args.vectors, args.layer, args.alpha, device)
    scorer = AnimalScorer(device)

    trial_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    configs = [
        (u, st, it)
        for u in args.refill_steps
        for st in args.sampling_temp
        for it in args.identify_temp
    ]
    for refill_steps, sampling_temp, identify_temp in configs:
        cfg_tag = f"u{refill_steps}_st{sampling_temp:g}_it{identify_temp:g}"
        records_path = args.output_dir / f"records_{cfg_tag}.jsonl"
        if records_path.is_file():
            with records_path.open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        else:
            records = []
            for source, target, direction_name in [
                ("cat", "dog", "cat_to_dog"),
                ("dog", "cat", "dog_to_cat"),
            ]:
                selected = [r for r in rows if r["concept"] == source]
                steer_vectors = {args.layer: base_vectors[direction_name]}
                for start in range(0, len(selected), args.steer_batch_size):
                    chunk = selected[start : start + args.steer_batch_size]
                    texts = [r["text"] for r in chunk]
                    inputs = tokenizer(
                        texts,
                        add_special_tokens=False,
                        padding=True,
                        truncation=True,
                        max_length=MAX_STEER_SEQ_LEN,
                        return_tensors="pt",
                    ).to(device)
                    with torch.no_grad():
                        step_results = resteer_v2(
                            model,
                            inputs,
                            steer_vectors,
                            resteer_steps=args.resteer_steps,
                            refill_steps=refill_steps,
                            sampling_temp=sampling_temp,
                            identify_temp=identify_temp,
                        )
                    for step in step_results:
                        k = int(step["resteer_step"]) + 1
                        decoded = tokenizer.batch_decode(
                            step["after"].to(device),
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        probs = scorer.target_probs(decoded, target=target, batch_size=args.eval_batch_size)
                        ppls = scorer.perplexities(decoded, batch_size=args.eval_batch_size)
                        for row, text, prob, ppl in zip(chunk, decoded, probs, ppls):
                            records.append(
                                {
                                    "stage": args.stage,
                                    "id": row["id"],
                                    "source_concept": source,
                                    "target_concept": target,
                                    "direction": direction_name,
                                    "length_bin": row["length_bin"],
                                    "word_count": row["word_count"],
                                    "k": k,
                                    "refill_steps": refill_steps,
                                    "sampling_temp": sampling_temp,
                                    "identify_temp": identify_temp,
                                    "original_text": row["text"],
                                    "steered_text": text.strip(),
                                    "target_prob": float(prob),
                                    "perplexity": float(ppl),
                                }
                            )
            with records_path.open("w", encoding="utf-8") as f:
                for row in records:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for k in range(1, args.resteer_steps + 1):
            subset = [r for r in records if int(r["k"]) == k]
            agg, scored = evaluate_records([dict(r) for r in subset])
            trial = {
                "stage": args.stage,
                "seed": args.seed,
                "vectors": str(args.vectors),
                "layer": args.layer,
                "alpha": args.alpha,
                "k": k,
                "refill_steps": refill_steps,
                "sampling_temp": sampling_temp,
                "identify_temp": identify_temp,
                **agg,
            }
            trial_rows.append(trial)
            for lr in aggregate_by_length(scored):
                length_rows.append({**trial, **lr})
            all_records.extend(scored)

    write_csv(args.output_dir / "asap_hyperparam_trials.csv", trial_rows)
    write_csv(args.output_dir / "asap_length_analysis.csv", length_rows)
    best = max(trial_rows, key=lambda r: float(r["harmonic_score"])) if trial_rows else {}
    qualitative = sorted(all_records, key=lambda r: float(r["harmonic_score"]), reverse=True)[:30]
    with (args.output_dir / "asap_best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with (args.output_dir / "asap_qualitative_examples.json").open("w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)
    print(json.dumps({"stage": args.stage, "best": best, "rows": len(trial_rows)}, indent=2))


if __name__ == "__main__":
    main()
