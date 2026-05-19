"""
Extract contrastive steering vectors (per-layer mean hidden states) from val_pos / val_neg
or from the fixed pair ("love", "hate") when --num-samples is 0.

Output format matches diffusion-get_steer.py: a dict with keys "positive" and "negative",
each a tuple of [num_layers+1] tensors of shape [hidden_dim].
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

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
        description="Extract diffusion steer vectors from val CSVs or love/hate (num-samples=0)."
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="0=only 'love' (pos) and 'hate' (neg); else first N rows from each val CSV (1, 5, 20, ...).",
    )
    args = ap.parse_args()

    n = args.num_samples
    if n == 0:
        pos_sample = list(["love"])
        neg_sample = list(["hate"])
        tag = "n0"
    else:
        pos_sample = load_texts_from_csv(Path("benchmarks/val_pos.csv"), n)
        neg_sample = load_texts_from_csv(Path("benchmarks/val_neg.csv"), n)
        tag = f"n{n}"

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
        f"layers={len(steer_vectors['positive'])})"
    )


if __name__ == "__main__":
    main()
