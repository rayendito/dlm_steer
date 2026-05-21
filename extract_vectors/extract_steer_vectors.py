"""
Extract contrastive steering vectors (per-layer mean hidden states).

Datasets (--data):
  imdb      — val_pos / val_neg under benchmarks/imdb/ (or love/hate when n=0)
  cats-dogs — train.csv: cat=positive, dog=negative (or cat/dog tokens when n=0)

Output: dict with keys "positive" and "negative", each a tuple of [num_layers+1]
tensors of shape [hidden_dim].
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
IMDB_VAL_POS = REPO_ROOT / "benchmarks/imdb/val_pos.csv"
IMDB_VAL_NEG = REPO_ROOT / "benchmarks/imdb/val_neg.csv"
CATS_DOGS_TRAIN = REPO_ROOT / "benchmarks/cats_dogs/train.csv"


def load_texts_from_csv(path: Path, n: int) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def load_texts_by_concept(path: Path, concept: str, n: int) -> list[str]:
    """First n non-empty rows where concept column matches (e.g. cat, dog)."""
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("concept") or "").strip().lower() != concept.lower():
                continue
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= n:
                break
    return texts


def resolve_samples(
    data: str,
    n: int,
) -> tuple[list[str], list[str], str]:
    if n == 0:
        if data == "cats-dogs":
            return ["cat"], ["dog"], "n0"
        return ["love"], ["hate"], "n0"

    if data == "imdb":
        pos = load_texts_from_csv(IMDB_VAL_POS, n)
        neg = load_texts_from_csv(IMDB_VAL_NEG, n)
        return pos, neg, f"n{n}"

    if data == "cats-dogs":
        pos = load_texts_by_concept(CATS_DOGS_TRAIN, "cat", n)
        neg = load_texts_by_concept(CATS_DOGS_TRAIN, "dog", n)
        return pos, neg, f"n{n}"

    raise ValueError(f"Unknown data: {data!r}")


def output_filename(data: str, tag: str) -> str:
    if data == "imdb":
        return f"diffusion-imdb-{tag}.pt"
    return f"diffusion-catdog-{tag}.pt"


def _mean_hidden_per_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    """Mask-mean over tokens, then mean over texts; processed in chunks to limit VRAM."""
    if not texts:
        raise ValueError("texts must be non-empty")

    layer_sums: list[torch.Tensor] | None = None
    n_seen = 0
    bs = max(1, batch_size)

    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(device)

        out = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1)

        per_instance = tuple(
            (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for h in out.hidden_states
        )
        if layer_sums is None:
            layer_sums = [h.sum(dim=0) for h in per_instance]
        else:
            for i, h in enumerate(per_instance):
                layer_sums[i] = layer_sums[i] + h.sum(dim=0)
        n_seen += len(chunk)
        del inputs, out, per_instance

    assert layer_sums is not None
    return tuple(s / n_seen for s in layer_sums)


@torch.no_grad()
def extract_steer_dict(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int = 8,
) -> dict[str, tuple[torch.Tensor, ...]]:
    steer_vectors: dict[str, tuple[torch.Tensor, ...]] = {}
    for sentiment, dataset in [("positive", pos_texts), ("negative", neg_texts)]:
        steer_vectors[sentiment] = _mean_hidden_per_layer(
            model, tokenizer, dataset, device, batch_size
        )
    return steer_vectors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract diffusion steer vectors from IMDB val or cats-dogs train CSVs."
    )
    ap.add_argument(
        "--data",
        choices=("imdb", "cats-dogs"),
        default="imdb",
        help="imdb: val_pos/val_neg; cats-dogs: cat=positive, dog=negative from train.csv.",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help=(
            "0=synthetic pair (love/hate for imdb, cat/dog for cats-dogs); "
            "else first N per class from val CSV (imdb) or train.csv (cats-dogs)."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Forward-pass batch size when averaging hidden states (lower if OOM).",
    )
    args = ap.parse_args()

    data = args.data
    n = args.num_samples
    pos_sample, neg_sample, tag = resolve_samples(data, n)

    if n > 0 and (not pos_sample or not neg_sample):
        raise SystemExit(
            f"Not enough texts for {data} with --num-samples {n}: "
            f"pos={len(pos_sample)} neg={len(neg_sample)}"
        )
    if n > 0 and (len(pos_sample) < n or len(neg_sample) < n):
        print(
            f"Warning: requested {n} per class, got pos={len(pos_sample)} neg={len(neg_sample)}",
            flush=True,
        )

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
        model,
        tokenizer,
        pos_sample,
        neg_sample,
        device,
        batch_size=max(1, args.batch_size),
    )

    out_path = REPO_ROOT / "steer_vectors" / output_filename(data, tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steer_vectors, out_path)
    pos_label = "cat" if data == "cats-dogs" else "pos"
    neg_label = "dog" if data == "cats-dogs" else "neg"
    print(
        f"Saved {out_path}  (data={data} {pos_label}={len(pos_sample)} {neg_label}={len(neg_sample)} texts, "
        f"layers={len(steer_vectors['positive'])})",
        flush=True,
    )


if __name__ == "__main__":
    main()
