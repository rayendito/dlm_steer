"""
Extract contrastive steering vectors as per-layer mean hidden states.

Supported datasets:
- imdb: positive/negative IMDB validation CSVs.
- cats-dogs: cats/dogs synthetic CSVs, with cat mapped to "positive" and dog
  mapped to "negative" so existing downstream code can keep using the same keys.

For cats-dogs, ``--num-samples 0`` uses token anchors instead of examples. With the
default ``--pair-order cat,dog``, positive=cat and negative=dog; use
``--pair-order dog,cat`` to flip that convention.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
IMDB_VAL_POS = REPO_ROOT / "benchmarks/imdb/val_pos.csv"
IMDB_VAL_NEG = REPO_ROOT / "benchmarks/imdb/val_neg.csv"
CATS_DOGS_TRAIN = REPO_ROOT / "benchmarks/cats_dogs/train.csv"
CATS_DOGS_VAL = REPO_ROOT / "benchmarks/cats_dogs/val.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def _sample(texts: list[str], n: int, seed: int) -> list[str]:
    texts = [t.strip() for t in texts if t and t.strip()]
    if n <= 0 or n >= len(texts):
        return texts
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    return [texts[i] for i in idx[:n]]


def load_texts_from_csv(path: Path, n: int, *, seed: int) -> list[str]:
    texts = [(r.get("text") or "").strip() for r in _read_rows(path)]
    return _sample(texts, n, seed)


def load_concept_texts_from_csv(path: Path, concept: str, n: int, *, seed: int) -> list[str]:
    concept = concept.strip().lower()
    texts = []
    for row in _read_rows(path):
        if (row.get("concept") or "").strip().lower() == concept:
            texts.append((row.get("text") or "").strip())
    return _sample(texts, n, seed)


def resolve_samples(
    data: str,
    n: int,
    *,
    seed: int,
    source_split: str,
    bench_dir: Path,
    pair_order: str,
) -> tuple[list[str], list[str], str]:
    if n == 0:
        if data == "cats-dogs":
            first, second = pair_order.split(",", 1)
            return [first.strip()], [second.strip()], "n0"
        return ["love"], ["hate"], "n0"

    if data == "imdb":
        pos = load_texts_from_csv(IMDB_VAL_POS, n, seed=seed)
        neg = load_texts_from_csv(IMDB_VAL_NEG, n, seed=seed)
        return pos, neg, f"n{n}"

    if data == "cats-dogs":
        path = bench_dir / "cats_dogs" / f"{source_split}.csv"
        if not path.is_file():
            path = CATS_DOGS_TRAIN if source_split == "train" else CATS_DOGS_VAL
        first, second = pair_order.split(",", 1)
        pos = load_concept_texts_from_csv(path, first.strip(), n, seed=seed)
        neg = load_concept_texts_from_csv(path, second.strip(), n, seed=seed)
        return pos, neg, f"{source_split}_n{n}_s{seed}"

    raise ValueError(f"Unknown data: {data!r}")


def output_filename(data: str, tag: str, pair_order: str) -> str:
    if data == "imdb":
        return f"diffusion-imdb-{tag}.pt"
    first, second = [x.strip() for x in pair_order.split(",", 1)]
    return f"diffusion-val-catdog_{first}_minus_{second}_{tag}.pt"


def _mean_hidden_per_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
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
            layer_sums = [h.sum(dim=0).detach().cpu() for h in per_instance]
        else:
            for i, h in enumerate(per_instance):
                layer_sums[i] = layer_sums[i] + h.sum(dim=0).detach().cpu()
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
    return {
        "positive": _mean_hidden_per_layer(model, tokenizer, pos_texts, device, batch_size),
        "negative": _mean_hidden_per_layer(model, tokenizer, neg_texts, device, batch_size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract diffusion steer vectors.")
    ap.add_argument(
        "--data",
        choices=("imdb", "cats-dogs"),
        default=None,
        help="imdb uses val_pos/val_neg; cats-dogs maps cat/dog concepts to positive/negative.",
    )
    ap.add_argument(
        "--concept-pair",
        choices=("sentiment", "cat-dog"),
        default=None,
        help="Backward-compatible alias: sentiment=imdb, cat-dog=cats-dogs.",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="0=token anchors; otherwise sample N rows per class/concept.",
    )
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--source-split", choices=("val", "train"), default="val")
    ap.add_argument("--bench-dir", type=Path, default=REPO_ROOT / "benchmarks")
    ap.add_argument(
        "--pair-order",
        type=str,
        default="cat,dog",
        help="Only for cats-dogs: positive,negative concept order. Use cat,dog or dog,cat.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("steer_vectors"))
    args = ap.parse_args()

    data = args.data
    if data is None:
        data = "cats-dogs" if args.concept_pair == "cat-dog" else "imdb"
    if args.concept_pair is not None:
        expected = "cats-dogs" if args.concept_pair == "cat-dog" else "imdb"
        if data != expected:
            raise ValueError("--data and --concept-pair disagree")

    if data == "cats-dogs":
        concepts = [x.strip().lower() for x in args.pair_order.split(",")]
        if concepts not in (["cat", "dog"], ["dog", "cat"]):
            raise ValueError("--pair-order must be 'cat,dog' or 'dog,cat'")

    pos_sample, neg_sample, tag = resolve_samples(
        data,
        args.num_samples,
        seed=args.sample_seed,
        source_split=args.source_split,
        bench_dir=args.bench_dir,
        pair_order=args.pair_order,
    )
    if not pos_sample or not neg_sample:
        raise RuntimeError(
            "Empty positive/negative sample after filtering/sampling: "
            f"positive={len(pos_sample)}, negative={len(neg_sample)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "GSAI-ML/LLaDA-8B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    steer_vectors = extract_steer_dict(
        model,
        tokenizer,
        pos_sample,
        neg_sample,
        device,
        batch_size=args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / output_filename(data, tag, args.pair_order)
    torch.save(steer_vectors, out_path)
    print(
        f"Saved {out_path} with positive={len(pos_sample)}, negative={len(neg_sample)}, "
        f"layers={len(steer_vectors['positive'])}, data={data}"
    )


if __name__ == "__main__":
    main()
