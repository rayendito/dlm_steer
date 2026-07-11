from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def read_labeled_rows(csv_path: Path) -> tuple[list[str], list[str]]:
    cat_texts: list[str] = []
    dog_texts: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            concept = (row.get("concept") or "").strip().lower()
            if not text or concept not in {"cat", "dog"}:
                continue
            if concept == "cat":
                cat_texts.append(text)
            else:
                dog_texts.append(text)
    return cat_texts, dog_texts


@torch.no_grad()
def extract_mean_vectors(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    max_length: int,
    batch_size: int,
) -> tuple[torch.Tensor, ...]:
    accum = None
    count = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        out = model(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].unsqueeze(-1)
        token_avg = tuple((h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) for h in out.hidden_states)
        layer_sums = tuple(t.sum(dim=0) for t in token_avg)
        if accum is None:
            accum = [x.clone() for x in layer_sums]
        else:
            for j in range(len(accum)):
                accum[j] += layer_sums[j]
        count += len(batch)
    if accum is None or count == 0:
        raise RuntimeError("No texts were available for vector extraction.")
    return tuple(x / count for x in accum)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract cats/dogs steering vectors from processed dataset split.")
    ap.add_argument("--train-csv", type=Path, default=Path("benchmarks/cats_dogs/train.csv"))
    ap.add_argument("--model-name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    ap.add_argument("--output-path", type=Path, default=Path("steer_vectors/cats_dogs.pt"))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cat_texts, dog_texts = read_labeled_rows(args.train_csv)
    if not cat_texts or not dog_texts:
        raise RuntimeError("Need both cat and dog rows in train split.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    cat_vecs = extract_mean_vectors(model, tokenizer, cat_texts, device, args.max_length, args.batch_size)
    dog_vecs = extract_mean_vectors(model, tokenizer, dog_texts, device, args.max_length, args.batch_size)

    payload = {"cat": cat_vecs, "dog": dog_vecs}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(
        f"Saved {args.output_path} with {len(cat_vecs)} layers "
        f"(cat={len(cat_texts)} rows, dog={len(dog_texts)} rows)."
    )


if __name__ == "__main__":
    main()
