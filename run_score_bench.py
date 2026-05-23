import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import eval_utils


EVAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score benchmark CSV rows with label probabilities and perplexity."
    )
    parser.add_argument(
        "--input_csv",
        "--input",
        "-i",
        type=Path,
        required=True,
        help="CSV to score.",
    )
    parser.add_argument(
        "--output_csv",
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Where to write the scored CSV.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of rows to score per model batch.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help='Column to score. Defaults to "output" if present, then "text".',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=("auto", "imdb", "catdog", "cats_dogs", "cats-dogs"),
        help="Label pair to score. Auto infers from path and CSV columns.",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default=EVAL_MODEL,
        help="Hugging Face causal LM used for scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device. Defaults to "cuda" when available, otherwise "cpu".',
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_text_column(df, requested):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"Requested text column {requested!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    for candidate in ("output", "text"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        'Could not infer text column. Pass --text_column; expected "output" or "text".'
    )


def infer_labels(df, input_csv, dataset):
    dataset_key = dataset.lower().replace("-", "_")
    path_key = input_csv.as_posix().lower().replace("-", "_")

    if dataset_key in {"catdog", "cats_dogs"}:
        return "dog", "cat"
    if dataset_key == "imdb":
        return "positive", "negative"

    concepts = set()
    if "concept" in df.columns:
        concepts = {
            str(value).strip().lower()
            for value in df["concept"].dropna().unique().tolist()
        }

    if {"cat", "dog"} & concepts or "catdog" in path_key or "cats_dogs" in path_key:
        return "dog", "cat"
    if {"positive", "negative"} & concepts or "imdb" in path_key:
        return "positive", "negative"

    raise ValueError(
        "Could not infer dataset labels. Pass --dataset imdb or --dataset catdog."
    )


def load_eval_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def score_csv(df, text_column, label_1, label_2, model, tokenizer, batch_size):
    texts = df[text_column].fillna("").astype(str).tolist()
    perplexities = []
    predicted_labels = []
    label_1_probs = []
    label_2_probs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch = texts[start:start + batch_size]
        classification, label_probs = eval_utils.score_labels(
            model,
            tokenizer,
            batch,
            label_1,
            label_2,
        )
        batch_perplexities = eval_utils.perplexity(model, tokenizer, batch)

        predicted_labels.extend(classification)
        label_1_probs.extend(float(prob) for prob in label_probs[label_1])
        label_2_probs.extend(float(prob) for prob in label_probs[label_2])
        perplexities.extend(float(ppl) for ppl in batch_perplexities)

    scored = df.copy()
    scored["perplexity"] = perplexities
    scored["predicted_label"] = predicted_labels
    scored[label_1] = label_1_probs
    scored[label_2] = label_2_probs
    return scored


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    df = pd.read_csv(args.input_csv)
    text_column = infer_text_column(df, args.text_column)
    label_1, label_2 = infer_labels(df, args.input_csv, args.dataset)

    device = resolve_device(args.device)
    eval_utils.device = device
    model, tokenizer = load_eval_model(args.eval_model, device)

    scored = score_csv(
        df,
        text_column,
        label_1,
        label_2,
        model,
        tokenizer,
        args.batch_size,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    print(
        f"Saved {len(scored)} scored rows to {args.output_csv} "
        f"(text_column={text_column}, labels={label_1}/{label_2})"
    )


if __name__ == "__main__":
    main()
