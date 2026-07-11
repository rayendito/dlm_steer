from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.generate import resteer_v2
from timpateks.llada.modeling_llada import LLaDAModelLM


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def harmonic(a: float, b: float, eps: float = 1e-12) -> float:
    return (2.0 * a * b) / (a + b + eps)


def robust_inv_ppl(ppls: list[float]) -> list[float]:
    vals = np.array(ppls, dtype=float)
    finite = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=float)
    if not finite.any():
        return out.tolist()
    lo = np.percentile(vals[finite], 5)
    hi = np.percentile(vals[finite], 95)
    clipped = np.clip(vals, lo, hi)
    cmin, cmax = clipped.min(), clipped.max()
    if cmax > cmin:
        norm = (clipped - cmin) / (cmax - cmin)
    else:
        norm = np.zeros_like(clipped)
    out = 1.0 - norm
    return out.tolist()


@dataclass
class Row:
    row_id: str
    text: str
    concept: str
    length_bin: int


def read_rows(csv_path: Path, max_rows: int | None = None) -> list[dict]:
    out: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            concept = (row.get("concept") or "").strip().lower()
            if not text or concept not in {"cat", "dog"}:
                continue
            out.append({"id": row.get("id", ""), "text": text, "concept": concept})
    if max_rows is not None:
        out = out[:max_rows]
    return out


def assign_length_bins(rows: list[dict]) -> list[Row]:
    lengths = np.array([len(r["text"].split()) for r in rows], dtype=float)
    q1, q2 = np.quantile(lengths, [1 / 3, 2 / 3])
    out: list[Row] = []
    for r, L in zip(rows, lengths):
        if L <= q1:
            b = 0
        elif L <= q2:
            b = 1
        else:
            b = 2
        out.append(Row(row_id=str(r["id"]), text=r["text"], concept=r["concept"], length_bin=b))
    return out


class QwenAnimalScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)
        self.cat_id = self.tokenizer.encode(" cat", add_special_tokens=False)[0]
        self.dog_id = self.tokenizer.encode(" dog", add_special_tokens=False)[0]

    @torch.no_grad()
    def target_probs(self, texts: list[str], target: str) -> list[float]:
        prompts = [f"Text: {t}\nAnimal:" for t in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs).logits
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        next_logits = logits[torch.arange(len(texts), device=self.device), last_idx]
        probs = F.softmax(next_logits, dim=-1)
        tid = self.dog_id if target == "dog" else self.cat_id
        return probs[:, tid].detach().cpu().tolist()

    @torch.no_grad()
    def perplexities(self, texts: list[str]) -> list[float]:
        raw = [("" if t is None else str(t)) for t in texts]
        safe = [t if t.strip() else "." for t in raw]
        enc = self.tokenizer(safe, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**enc)
        shift_logits = out.logits[:, :-1]
        shift_labels = enc["input_ids"][:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)
        mask = enc["attention_mask"][:, 1:]
        denom = mask.sum(dim=1).clamp(min=1)
        loss = (loss * mask).sum(dim=1) / denom
        ppl = torch.exp(loss).detach().cpu().tolist()
        return ppl


class LladaResteerer:
    def __init__(self, model_name: str, vectors_path: Path, steer_layers: list[int], steer_alpha: float) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = LLaDAConfig.from_pretrained(model_name)
        self.model = LLaDAModelLM.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        vecs = torch.load(vectors_path, map_location=self.device)
        cat_vecs = vecs["cat"]
        dog_vecs = vecs["dog"]
        self.steer_layers = steer_layers
        self.steer_alpha = steer_alpha
        self.base_cat_to_dog = {
            li: steer_alpha * (l2_normalize(dog_vecs[li].to(self.device)) - l2_normalize(cat_vecs[li].to(self.device)))
            for li in steer_layers
        }
        self.base_dog_to_cat = {li: -v for li, v in self.base_cat_to_dog.items()}

    @torch.no_grad()
    def steer_batch(
        self,
        texts: list[str],
        direction: str,
        identify_temp: float,
        resteer_steps: int,
        refill_steps: int,
        sampling_temp: float = 1.0,
    ) -> list[str]:
        inputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        steer_vectors = self.base_cat_to_dog if direction == "cat_to_dog" else self.base_dog_to_cat
        _ = resteer_v2(
            self.model,
            tokenized_inputs=inputs,
            steer_vectors=steer_vectors,
            resteer_steps=resteer_steps,
            refill_steps=refill_steps,
            sampling_temp=sampling_temp,
            identify_temp=identify_temp,
            alpha_decay=False,
        )
        decoded = self.tokenizer.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [d.strip() for d in decoded]


def batched(xs: list, bs: int) -> Iterable[list]:
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


def evaluate_setup(
    rows: list[Row],
    resteerer: LladaResteerer,
    scorer: QwenAnimalScorer,
    length_bin: int,
    identify_temp: float,
    resteer_steps: int,
    refill_steps: int,
    batch_size: int,
) -> dict:
    selected = [r for r in rows if r.length_bin == length_bin]
    all_records: list[dict] = []
    for direction in ("cat_to_dog", "dog_to_cat"):
        if direction == "cat_to_dog":
            src = [r for r in selected if r.concept == "cat"]
            target = "dog"
        else:
            src = [r for r in selected if r.concept == "dog"]
            target = "cat"
        for chunk in batched(src, batch_size):
            orig_texts = [r.text for r in chunk]
            steered = resteerer.steer_batch(
                orig_texts,
                direction=direction,
                identify_temp=identify_temp,
                resteer_steps=resteer_steps,
                refill_steps=refill_steps,
            )
            tgt = scorer.target_probs(steered, target=target)
            ppl = scorer.perplexities(steered)
            for i, row in enumerate(chunk):
                all_records.append(
                    {
                        "id": row.row_id,
                        "direction": direction,
                        "source_concept": row.concept,
                        "target_concept": target,
                        "length_bin": row.length_bin,
                        "original_text": row.text,
                        "steered_text": steered[i],
                        "target_prob": float(tgt[i]),
                        "perplexity": float(ppl[i]),
                    }
                )
    if not all_records:
        return {
            "target_prob": 0.0,
            "perplexity": float("inf"),
            "harmonic_score": 0.0,
            "records": [],
        }
    ppls = [r["perplexity"] for r in all_records]
    inv_ppl = robust_inv_ppl(ppls)
    hm_scores: list[float] = []
    for i, r in enumerate(all_records):
        hm = harmonic(r["target_prob"], inv_ppl[i])
        hm_scores.append(hm)
        r["inv_ppl_norm"] = inv_ppl[i]
        r["harmonic_score"] = hm
    return {
        "target_prob": float(np.mean([r["target_prob"] for r in all_records])),
        "perplexity": float(np.mean(ppls)),
        "harmonic_score": float(np.mean(hm_scores)),
        "records": all_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy ablation for cats-vs-dogs TIMPA steering.")
    ap.add_argument("--dataset-csv", type=Path, default=Path("benchmarks/cats_dogs/test.csv"))
    ap.add_argument("--vectors-path", type=Path, default=Path("steer_vectors/cats_dogs.pt"))
    ap.add_argument("--llada-model", type=str, default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--output-dir", type=Path, default=Path("cats_dogs/results"))
    ap.add_argument("--steer-layers", type=int, nargs="+", default=[25, 31, 16])
    ap.add_argument("--steer-alpha", type=float, default=50.0)
    ap.add_argument("--temperature-grid", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--u-grid", type=int, nargs="+", default=[1, 3, 5, 10, 15, 25])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_rows = read_rows(args.dataset_csv, max_rows=args.max_rows)
    rows = assign_length_bins(base_rows)
    resteerer = LladaResteerer(
        model_name=args.llada_model,
        vectors_path=args.vectors_path,
        steer_layers=args.steer_layers,
        steer_alpha=args.steer_alpha,
    )
    scorer = QwenAnimalScorer()

    best = {"N": 0, "T": 1e-4, "k": 1, "u": 1}
    all_trials: list[dict] = []

    # Stage N
    for N in [0, 1, 2]:
        res = evaluate_setup(rows, resteerer, scorer, N, best["T"], best["k"], best["u"], args.batch_size)
        trial = {"stage": "N", "N": N, "T": best["T"], "k": best["k"], "u": best["u"], **{k: res[k] for k in ("target_prob", "perplexity", "harmonic_score")}}
        all_trials.append(trial)
    best_N = max([t for t in all_trials if t["stage"] == "N"], key=lambda x: x["harmonic_score"])
    best["N"] = int(best_N["N"])

    # Stage T
    for T in args.temperature_grid:
        res = evaluate_setup(rows, resteerer, scorer, best["N"], float(T), best["k"], best["u"], args.batch_size)
        all_trials.append({"stage": "T", "N": best["N"], "T": float(T), "k": best["k"], "u": best["u"], **{k: res[k] for k in ("target_prob", "perplexity", "harmonic_score")}})
    best_T = max([t for t in all_trials if t["stage"] == "T"], key=lambda x: x["harmonic_score"])
    best["T"] = float(best_T["T"])

    # Stage k
    for k in args.k_grid:
        res = evaluate_setup(rows, resteerer, scorer, best["N"], best["T"], int(k), best["u"], args.batch_size)
        all_trials.append({"stage": "k", "N": best["N"], "T": best["T"], "k": int(k), "u": best["u"], **{kk: res[kk] for kk in ("target_prob", "perplexity", "harmonic_score")}})
    best_k = max([t for t in all_trials if t["stage"] == "k"], key=lambda x: x["harmonic_score"])
    best["k"] = int(best_k["k"])

    # Stage u
    final_records: list[dict] = []
    for u in args.u_grid:
        res = evaluate_setup(rows, resteerer, scorer, best["N"], best["T"], best["k"], int(u), args.batch_size)
        all_trials.append({"stage": "u", "N": best["N"], "T": best["T"], "k": best["k"], "u": int(u), **{kk: res[kk] for kk in ("target_prob", "perplexity", "harmonic_score")}})
        if int(u) == args.u_grid[0]:
            final_records = res["records"]
    best_u = max([t for t in all_trials if t["stage"] == "u"], key=lambda x: x["harmonic_score"])
    best["u"] = int(best_u["u"])

    # Recompute final best for record dump
    final_eval = evaluate_setup(rows, resteerer, scorer, best["N"], best["T"], best["k"], best["u"], args.batch_size)
    final_records = final_eval["records"]
    qual = sorted(final_records, key=lambda x: x["harmonic_score"], reverse=True)[:20]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "greedy_trials.json").open("w", encoding="utf-8") as f:
        json.dump(all_trials, f, indent=2)
    with (args.output_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "final_metrics": {k: final_eval[k] for k in ("target_prob", "perplexity", "harmonic_score")}}, f, indent=2)
    with (args.output_dir / "qualitative_examples.json").open("w", encoding="utf-8") as f:
        json.dump(qual, f, indent=2)

    csv_path = args.output_dir / "greedy_trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["stage", "N", "T", "k", "u", "target_prob", "perplexity", "harmonic_score"]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(all_trials)

    print(json.dumps({"best": best, "results_csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
