"""
Extract contrastive steering vectors (per-layer mean hidden states).

Supports:
- Sentiment mode (positive/negative).
- Cat-vs-dog mode.

Output format matches existing downstream code: dict with keys "positive"/"negative",
each containing [num_layers+1] tensors of shape [hidden_dim].
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def _read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def load_texts_from_csv(path: Path, n: int, *, seed: int) -> list[str]:

    rows = _read_rows(path)
    texts = [(r.get("text") or "").strip() for r in rows]
    texts = [t for t in texts if t]
    if n <= 0 or n >= len(texts):
        return texts
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx[:n]]
    return texts


def load_concept_texts_from_csv(path: Path, concept: str, n: int, *, seed: int) -> list[str]:
    rows = _read_rows(path)
    concept = concept.strip().lower()
    texts = []
    for row in rows:
        c = (row.get("concept") or "").strip().lower()
        t = (row.get("text") or "").strip()
        if c == concept and t:
            texts.append(t)
    if n <= 0 or n >= len(texts):
        return texts
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx[:n]]
    return texts

@torch.no_grad()
def extract_steer_dict(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
) -> dict[str, tuple[torch.Tensor, ...]]:
    steer_vectors: dict[str, tuple[torch.Tensor, ...]] = {}
    for sentiment, dataset in [("positive", pos_texts), ("negative", neg_texts)]:
        inputs = tokenizer(
            dataset,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)

        out = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1)

        averaged_over_tokens = tuple(
            (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for h in out.hidden_states
        )
        averaged_over_instances = tuple(
            h.mean(dim=0) for h in averaged_over_tokens
        )
        steer_vectors[sentiment] = averaged_over_instances
    return steer_vectors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract diffusion steer vectors for sentiment or cat-dog concepts."
    )
    ap.add_argument(
        "--concept-pair",
        type=str,
        default="sentiment",
        choices=["sentiment", "cat-dog"],
        help="Which contrastive concept pair to extract vectors for.",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="0=use concept tokens only; else sample N rows per class from selected split.",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling rows when num-samples > 0.",
    )
    ap.add_argument(
        "--source-split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Split used to construct steer vectors.",
    )
    ap.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory containing benchmark CSV files.",
    )
    ap.add_argument(
        "--pair-order",
        type=str,
        default="cat,dog",
        help="Only for cat-dog: positive,negative concept order. Example: cat,dog or dog,cat.",
    )
    args = ap.parse_args()

    n = args.num_samples
    if args.concept_pair == "sentiment":
        if n == 0:
            pos_sample = ["love"]
            neg_sample = ["hate"]
            tag = "sentiment_n0"
        else:
            pos_csv = args.bench_dir / f"{args.source_split}_pos.csv"
            neg_csv = args.bench_dir / f"{args.source_split}_neg.csv"
            pos_sample = load_texts_from_csv(pos_csv, n, seed=args.sample_seed)
            neg_sample = load_texts_from_csv(neg_csv, n, seed=args.sample_seed + 1)
            tag = f"sentiment_{args.source_split}_n{n}_s{args.sample_seed}"
    else:
        order = [x.strip().lower() for x in args.pair_order.split(",")]
        if len(order) != 2 or set(order) != {"cat", "dog"}:
            raise ValueError("--pair-order must be 'cat,dog' or 'dog,cat'")
        pos_concept, neg_concept = order[0], order[1]
        if n == 0:
            pos_sample = [pos_concept]
            neg_sample = [neg_concept]
            tag = f"catdog_{pos_concept}_minus_{neg_concept}_n0"
        else:
            concept_csv = args.bench_dir / "cats_dogs" / f"{args.source_split}.csv"
            pos_sample = load_concept_texts_from_csv(
                concept_csv, pos_concept, n, seed=args.sample_seed
            )
            neg_sample = load_concept_texts_from_csv(
                concept_csv, neg_concept, n, seed=args.sample_seed + 1
            )
            tag = (
                f"catdog_{pos_concept}_minus_{neg_concept}_"
                f"{args.source_split}_n{n}_s{args.sample_seed}"
            )

    if not pos_sample or not neg_sample:
        raise RuntimeError("Empty positive/negative sample after filtering/sampling.")

    device = "cuda"
    torch.cuda.empty_cache() if device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    steer_vectors = extract_steer_dict(
        model, tokenizer, pos_sample, neg_sample, device
    )

    out_path = Path("steer_vectors") / f"diffusion-val-{tag}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steer_vectors, out_path)
    print(
        f"Saved {out_path}  (pos={len(pos_sample)} neg={len(neg_sample)} texts, "
        f"layers={len(steer_vectors['positive'])}, concept_pair={args.concept_pair})"
    )


if __name__ == "__main__":
    main()
