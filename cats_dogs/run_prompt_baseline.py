#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llada.configuration_llada import LLaDAConfig
from llada.generate import generate
from llada.modeling_llada import LLaDAModelLM


LLADA_ID = "GSAI-ML/LLaDA-8B-Base"
QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MASK_ID = 126336


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


def read_cats_dogs(path: Path, max_per_class: int | None, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            concept = (row.get("concept") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if concept in {"cat", "dog"} and text:
                rows.append({"id": row.get("id") or str(i), "label": concept, "text": text})
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in ("cat", "dog"):
        subset = [r for r in rows if r["label"] == label]
        rng.shuffle(subset)
        if max_per_class is not None:
            subset = subset[:max_per_class]
        out.extend(subset)
    rng.shuffle(out)
    return out


def read_imdb_dir(path: Path, max_per_class: int | None, seed: int) -> list[dict[str, Any]]:
    candidate_groups = [
        [(path / "train_pos.csv", "positive"), (path / "train_neg.csv", "negative")],
        [(path / "val_pos.csv", "positive"), (path / "val_neg.csv", "negative")],
        [(path / "positive.txt", "positive"), (path / "negative.txt", "negative")],
        [(path / "pos.txt", "positive"), (path / "neg.txt", "negative")],
    ]
    rows: list[dict[str, Any]] = []
    candidates = next((g for g in candidate_groups if all(p.is_file() for p, _ in g)), [])
    for file_path, label in candidates:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".csv":
            with file_path.open(encoding="utf-8", errors="ignore", newline="") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    text = (row.get("text") or "").strip()
                    if text:
                        rows.append({"id": row.get("id") or f"{file_path.stem}-{i}", "label": label, "text": text})
        else:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    text = line.strip()
                    if text:
                        rows.append({"id": f"{file_path.stem}-{i}", "label": label, "text": text})
    if not rows:
        raise FileNotFoundError(f"No IMDB positive/negative text files found under {path}")
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in ("positive", "negative"):
        subset = [r for r in rows if r["label"] == label]
        rng.shuffle(subset)
        if max_per_class is not None:
            subset = subset[:max_per_class]
        out.extend(subset)
    rng.shuffle(out)
    return out


def build_prompt(task: str, source: str, text: str) -> tuple[str, str, str]:
    if task == "cats_dogs":
        target = "dog" if source == "cat" else "cat"
        prompt = (
            f"Change this sentence to a sentence about {target}s while preserving the original meaning.\n"
            f"Sentence: {text}\nRewritten sentence:"
        )
        return target, f"{source}_to_{target}", prompt
    target = "negative" if source == "positive" else "positive"
    prompt = (
        f"Change this movie review to a {target} review while preserving the original content.\n"
        f"Review: {text}\nRewritten review:"
    )
    return target, f"{source}_to_{target}", prompt


def clean_generation(prompt: str, decoded: str) -> str:
    text = decoded
    if text.startswith(prompt):
        text = text[len(prompt) :]
    for marker in ["Rewritten sentence:", "Rewritten review:"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    text = text.replace("<|endoftext|>", " ").replace("[PAD]", " ").replace("[MASK]", " ")
    text = " ".join(text.split())
    return text.strip(" \n\t\"'")


class QwenScorer:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_ID).eval().to(device)
        self.ids = {
            "cat": self.tokenizer.encode(" cat", add_special_tokens=False)[0],
            "dog": self.tokenizer.encode(" dog", add_special_tokens=False)[0],
            "positive": self.tokenizer.encode(" positive", add_special_tokens=False)[0],
            "negative": self.tokenizer.encode(" negative", add_special_tokens=False)[0],
        }

    @torch.no_grad()
    def target_probs(self, texts: list[str], targets: list[str], task: str, batch_size: int) -> list[float]:
        out: list[float] = []
        field = "Animal" if task == "cats_dogs" else "Sentiment"
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            target_chunk = targets[i : i + batch_size]
            prompts = [f"Text: {t}\n{field}:" for t in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inputs).logits
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            next_logits = logits[torch.arange(len(chunk), device=self.device), last_idx]
            probs = F.softmax(next_logits, dim=-1)
            for j, target in enumerate(target_chunk):
                out.append(float(probs[j, self.ids[target]].detach().cpu()))
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


def tfidf_similarity(originals: list[str], rewrites: list[str]) -> list[float]:
    if not originals:
        return []
    vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vect.fit_transform(originals + rewrites)
    left = matrix[: len(originals)]
    right = matrix[len(originals) :]
    sims = cosine_similarity(left, right).diagonal()
    return [float(max(0.0, min(1.0, x))) for x in sims]


def summarize(records: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({r["direction"] for r in records}) + ["overall"]
    for group in groups:
        subset = records if group == "overall" else [r for r in records if r["direction"] == group]
        if not subset:
            continue
        finite_ppl = [float(r["perplexity"]) for r in subset if math.isfinite(float(r["perplexity"]))]
        rows.append(
            {
                "task": task,
                "method": "prompting",
                "direction": group,
                "num_records": len(subset),
                "target_prob": float(np.mean([float(r["target_prob"]) for r in subset])),
                "perplexity": float(np.mean(finite_ppl)) if finite_ppl else float("inf"),
                "harmonic_score": float(np.mean([float(r["harmonic_score"]) for r in subset])),
                "semantic_similarity": float(np.mean([float(r["semantic_similarity"]) for r in subset])),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cats_dogs", "imdb"], required=True)
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--max-prompt-length", type=int, default=896)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=128)
    ap.add_argument("--sampling-temp", type=float, default=0.5)
    args = ap.parse_args()

    if args.gen_length % args.block_length != 0 or args.steps % (args.gen_length // args.block_length) != 0:
        raise ValueError("--gen-length must be divisible by --block-length and compatible with --steps")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = (
        read_cats_dogs(args.dataset_path, args.max_per_class, args.seed)
        if args.task == "cats_dogs"
        else read_imdb_dir(args.dataset_path, args.max_per_class, args.seed)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / f"{args.task}_prompt_records.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LLaDAConfig.from_pretrained(LLADA_ID)
    model = LLaDAModelLM.from_pretrained(LLADA_ID, config=cfg, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(LLADA_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"

    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        targets, directions, prompts = [], [], []
        for row in chunk:
            target, direction, prompt = build_prompt(args.task, row["label"], row["text"])
            targets.append(target)
            directions.append(direction)
            prompts.append(prompt)
        enc = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = generate(
                model,
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.sampling_temp,
                remasking="low_confidence",
                mask_id=MASK_ID,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for row, target, direction, prompt, text in zip(chunk, targets, directions, prompts, decoded):
            records.append(
                {
                    "task": args.task,
                    "id": row["id"],
                    "source_label": row["label"],
                    "target_label": target,
                    "direction": direction,
                    "original_text": row["text"],
                    "generated_text": clean_generation(prompt, text),
                }
            )
        with records_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    scorer = QwenScorer(device)
    generated = [r["generated_text"] for r in records]
    target_probs = scorer.target_probs(generated, [r["target_label"] for r in records], args.task, args.eval_batch_size)
    ppls = scorer.perplexities(generated, args.eval_batch_size)
    sims = tfidf_similarity([r["original_text"] for r in records], generated)
    invs = inv_ppl_scores(ppls)
    for row, prob, ppl, sim, inv in zip(records, target_probs, ppls, sims, invs):
        row["target_prob"] = float(prob)
        row["perplexity"] = float(ppl)
        row["inv_ppl_norm"] = float(inv)
        row["semantic_similarity"] = float(sim)
        row["harmonic_score"] = float(harmonic(float(prob), float(inv)))

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, args.task)
    write_csv(args.output_dir / f"{args.task}_prompt_summary.csv", summary)
    qualitative = sorted(records, key=lambda r: (float(r["harmonic_score"]), float(r["semantic_similarity"])), reverse=True)[:40]
    with (args.output_dir / f"{args.task}_prompt_qualitative_examples.json").open("w", encoding="utf-8") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)
    print(json.dumps({"task": args.task, "records": len(records), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
