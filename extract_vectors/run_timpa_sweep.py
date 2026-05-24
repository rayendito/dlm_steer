#!/usr/bin/env python3
"""
TIMPA-style steering sweep (generation only) for ``extract_vectors/``.

Runs ``resteer_v2`` over val prompts and writes ``scores.json``. Score and plot
heatmaps separately with ``extract_vectors/score_timpa_sweep.py``.

Run from repo root::

    python extract_vectors/run_timpa_sweep.py \\
        --direction negative \\
        --vectors steer_vectors/diffusion-catdog-n100.pt

    python extract_vectors/score_timpa_sweep.py \\
        --results-dir extract_vectors/results_timpa/results_neg_100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llada.configuration_llada import LLaDAConfig
from llada.generate import resteer_v2
from llada.modeling_llada import LLaDAModelLM

# --- hardcoded sweep / resteer ------------------------------------------------
IMDB_VAL_POS = _REPO_ROOT / "benchmarks/imdb/val_pos.csv"
IMDB_VAL_NEG = _REPO_ROOT / "benchmarks/imdb/val_neg.csv"
CATS_DOGS_VAL = _REPO_ROOT / "benchmarks/cats_dogs/val.csv"

MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
DEVICE = "cuda"
SEED = 42
VAL_LIMIT = 20

ALPHAS = [10, 50, 100, 200, 500]

LAYER_MIN = 0
LAYER_MAX = 33
MAX_STEER_SEQ_LEN = 1024
RESTEER_STEPS = 15
REFILL_STEPS = 15
IDENTIFY_TEMP = 0.5
SAMPLING_TEMP = 0.5
RESTEER_BATCH_SIZE = 1

RESULTS_PARENT = Path("extract_vectors") / "results_timpa"


def l2_normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def results_tag_from_vectors_path(vectors_path: Path) -> str:
    stem = vectors_path.stem.lower()
    if "catdog" in stem:
        tag = re.sub(r"^diffusion[-_]?val[-_]?", "", stem)
        tag = re.sub(r"[^a-z0-9]+", "_", tag).strip("_")
        return tag
    for pat in (r"_n(\d+)$", r"-n(\d+)$", r"_n(\d+)", r"-n(\d+)"):
        m = re.search(pat, stem)
        if m:
            return m.group(1)
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:48]
    return slug if slug else "unknown"


def infer_data_from_vectors(vectors_path: Path, data_cli: str | None) -> str:
    if data_cli is not None:
        if data_cli not in ("imdb", "cats-dogs"):
            raise ValueError(f"--data must be imdb or cats-dogs, got {data_cli!r}")
        return data_cli
    if "catdog" in vectors_path.stem.lower():
        return "cats-dogs"
    return "imdb"


def _direction_results_abbrev(direction: str) -> str:
    if direction == "positive":
        return "pos"
    if direction == "negative":
        return "neg"
    raise ValueError(f"direction must be positive or negative, got {direction!r}")


def resolve_result_dir(vectors_path: Path, direction: str) -> tuple[Path, str]:
    tag = results_tag_from_vectors_path(vectors_path)
    abbrev = _direction_results_abbrev(direction)
    return RESULTS_PARENT / f"results_{abbrev}_{tag}", tag


def _load_cats_dogs_val_texts(path: Path, concept: str, limit: int) -> list[str]:
    concept = concept.strip().lower()
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("concept") or "").strip().lower() != concept:
                continue
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= limit:
                break
    return texts


def _load_imdb_val_texts(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= limit:
                break

    # DEBUG
    # texts = ["This movie is sucks so bad, action is poor, and the plot is stupid."]

    return texts


def load_val_prompt_texts(
    data: str, direction: str
) -> tuple[list[tuple[int, str, str]], str]:
    rows: list[tuple[int, str, str]] = []

    if data == "cats-dogs":
        if not CATS_DOGS_VAL.is_file():
            raise FileNotFoundError(f"Missing val CSV: {CATS_DOGS_VAL}")
        if direction == "positive":
            source_concept, label = "dog", "dog"
            desc = f"cats_dogs/val.csv dog rows (steer toward cat), n={VAL_LIMIT}"
        elif direction == "negative":
            source_concept, label = "cat", "cat"
            desc = f"cats_dogs/val.csv cat rows (steer toward dog), n={VAL_LIMIT}"
        else:
            raise ValueError("direction must be positive or negative")
        texts = _load_cats_dogs_val_texts(CATS_DOGS_VAL, source_concept, VAL_LIMIT)
    elif data == "imdb":
        if direction == "positive":
            path, label = IMDB_VAL_NEG, "neg"
            desc = f"imdb val_neg (steer toward positive), n={VAL_LIMIT}"
        elif direction == "negative":
            path, label = IMDB_VAL_POS, "pos"
            desc = f"imdb val_pos (steer toward negative), n={VAL_LIMIT}"
        else:
            raise ValueError("direction must be positive or negative")
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV: {path}")
        texts = _load_imdb_val_texts(path, VAL_LIMIT)
    else:
        raise ValueError(f"Unknown data: {data!r}")

    if not texts:
        raise ValueError(f"No val texts loaded for data={data!r} direction={direction!r}")

    for idx, text in enumerate(texts):
        rows.append((idx, text, label))
    return rows, desc


def build_steer_bases(
    vectors_path: Path,
    direction: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], int]:
    if not vectors_path.is_file():
        raise FileNotFoundError(f"Vectors not found: {vectors_path}")

    raw = torch.load(vectors_path, map_location=device)
    pos_vectors = raw["positive"]
    neg_vectors = raw["negative"]

    steer_all = tuple(
        l2_normalize(neg_vectors[i].to(dtype=dtype, device=device))
        - l2_normalize(pos_vectors[i].to(dtype=dtype, device=device))
        for i in range(len(pos_vectors))
    )
    if direction == "positive":
        steer_all = tuple(-v for v in steer_all)
    elif direction != "negative":
        raise ValueError("--direction must be positive or negative")
    return steer_all, len(steer_all)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TIMPA-style val steering sweep (generation only; L2 norm, resteer_v2)."
    )
    ap.add_argument(
        "--direction",
        choices=["positive", "negative"],
        required=True,
        help="Steering target. IMDB: positive/negative. cats-dogs: positive=cat, negative=dog.",
    )
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument(
        "--data",
        choices=["imdb", "cats-dogs"],
        default=None,
        help="Dataset (default: infer from --vectors filename).",
    )
    args = ap.parse_args()

    vectors_path = Path(args.vectors)
    data = infer_data_from_vectors(vectors_path, args.data)
    out_dir, vectors_tag = resolve_result_dir(vectors_path, args.direction)
    out_sweep_json = out_dir / "scores.json"
    print(
        f"Outputs → {out_dir} (data={data}, direction={args.direction}, tag={vectors_tag})",
        flush=True,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    val_rows, prompt_desc = load_val_prompt_texts(data, args.direction)
    texts = [t for _, t, _ in val_rows]
    prompt_idxs = [i for i, _, _ in val_rows]
    print(f"Val prompts: {len(val_rows)} rows — {prompt_desc}", flush=True)

    device = DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    config = LLaDAConfig.from_pretrained(MODEL_ID)
    model = LLaDAModelLM.from_pretrained(
        MODEL_ID,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    model_dtype = torch.bfloat16
    steer_bases, num_layers = build_steer_bases(
        vectors_path, args.direction, device, model_dtype
    )

    layer_lo = max(0, LAYER_MIN)
    layer_hi = min(LAYER_MAX, num_layers - 1)
    if layer_lo > layer_hi:
        raise SystemExit(
            f"Invalid layer range {LAYER_MIN}-{LAYER_MAX} (model has {num_layers} layers)"
        )

    results: list[dict[str, Any]] = []
    sweep_pairs = list(product(ALPHAS, range(layer_lo, layer_hi + 1)))

    for alpha, layer in tqdm(sweep_pairs, desc="resteer_v2 sweep (α × layer)", unit="pair"):
        base = steer_bases[layer]
        alpha_t = torch.tensor(alpha, dtype=base.dtype, device=base.device)
        steer_vectors = {layer: base * alpha_t}
        gb = max(1, RESTEER_BATCH_SIZE)
        for c in range(0, len(texts), gb):
            chunk_pidx = prompt_idxs[c : c + gb]
            chunk_texts = texts[c : c + gb]
            tokenized_inputs = tokenizer(
                chunk_texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=MAX_STEER_SEQ_LEN,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                resteer_v2(
                    model,
                    tokenized_inputs,
                    steer_vectors,
                    resteer_steps=RESTEER_STEPS,
                    refill_steps=REFILL_STEPS,
                    sampling_temp=SAMPLING_TEMP,
                    identify_temp=IDENTIFY_TEMP,
                    alpha_decay=False,
                )
            steered_ids = tokenized_inputs["input_ids"]
            decoded = tokenizer.batch_decode(
                steered_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for pidx, text, steered_text in zip(chunk_pidx, chunk_texts, decoded):
                results.append(
                    {
                        "prompt_idx": pidx,
                        "prompt": text,
                        "alpha": float(alpha),
                        "layer": layer,
                        "generation": steered_text.strip(),
                        "mode": "steered",
                    }
                )

    payload = {
        "data": data,
        "direction": args.direction,
        "vectors_file": str(vectors_path.resolve()),
        "vectors_results_tag": vectors_tag,
        "results_directory": str(out_dir.resolve()),
        "prompt_selection": prompt_desc,
        "val_limit": VAL_LIMIT,
        "steering_method": "resteer_v2",
        "steer_normalization": "l2",
        "model": MODEL_ID,
        "resteer_steps": RESTEER_STEPS,
        "refill_steps": REFILL_STEPS,
        "identify_temp": IDENTIFY_TEMP,
        "sampling_temp": SAMPLING_TEMP,
        "max_steer_seq_len": MAX_STEER_SEQ_LEN,
        "alphas": ALPHAS,
        "layer_min": layer_lo,
        "layer_max": layer_hi,
        "num_prompts": len(val_rows),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_sweep_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved sweep generations to {out_sweep_json}", flush=True)
    print("Run score_timpa_sweep.py on this directory to add scores and heatmaps.", flush=True)


if __name__ == "__main__":
    main()
