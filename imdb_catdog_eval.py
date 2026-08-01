#!/usr/bin/env python3
"""Evaluate full IMDb and Cat/Dog TIMPA-probabilistic test runs."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from timpa_eval import (
    eval_temp_classification,
    llmjudge_faithfulness,
    llmjudge_retain_structure,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


#### CLASSIFICATION MODEL
DEVICE = "cuda"
CLASSIFIER_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
LOCAL_FILES_ONLY = True
CLASSIFICATION_BATCH_SIZE = 5

#### LLM JUDGE
LLM_JUDGE_MODEL = "openai/gpt-4.1-mini"
LLM_JUDGE_BATCH_SIZE = 10
LLM_JUDGE_TIMEOUT = 300

#### INPUTS AND OUTPUT
RESULTS_ROOT = Path("timpateks_results")
DATASET_DIRECTIONS = {
    "imdb": ("positive", "negative"),
    "catdog": ("cat", "dog"),
}
EXPECTED_EXAMPLES_PER_DIRECTION = 100
OUTPUT_FILENAME = "evaluation.csv"
CLASSIFICATION_FILENAME = "classification_evaluation.csv"

# Keep the spelling requested for the output schema.
FIELDNAMES = [
    "sentence_id",
    "succesfully_steered",
    "target_probability_gain",
    "faithfulness_score",
    "structure_retention_score",
]


@dataclass(frozen=True)
class EvaluationItem:
    sentence_id: int
    target_direction: str
    text_before: str
    text_after: str


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full IMDb and Cat/Dog TIMPA-probabilistic test runs."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate trials that already have a complete evaluation.csv.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and count all inputs without loading models or calling APIs.",
    )
    parser.add_argument(
        "--classification-only",
        action="store_true",
        help=(
            "Run only Qwen classification and target-probability gain, "
            "saving resumable per-trial checkpoints."
        ),
    )
    return parser.parse_args()


def _read_before_after_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"text_before", "text_after"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing columns: {missing}.")
        rows = list(reader)

    if len(rows) != EXPECTED_EXAMPLES_PER_DIRECTION:
        raise ValueError(
            f"Expected {EXPECTED_EXAMPLES_PER_DIRECTION} examples in {path}, "
            f"found {len(rows)}."
        )
    if any(
        not row["text_before"].strip() or not row["text_after"].strip()
        for row in rows
    ):
        raise ValueError(f"{path} contains an empty before/after text.")
    return rows


def _find_trial_directories(dataset_name, directions):
    dataset_root = RESULTS_ROOT / f"{dataset_name}_test_csv"
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Full-test directory does not exist: {dataset_root}"
        )

    first_direction = directions[0]
    trial_directories = sorted(
        path.parent
        for path in dataset_root.glob(f"seed*/*/{first_direction}.csv")
    )
    if not trial_directories:
        raise FileNotFoundError(
            f"No completed {dataset_name} trials were found under {dataset_root}."
        )

    for trial_directory in trial_directories:
        missing = [
            f"{direction}.csv"
            for direction in directions
            if not (trial_directory / f"{direction}.csv").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{trial_directory} is missing required inputs: "
                f"{', '.join(missing)}."
            )
    return dataset_root, trial_directories


def _read_trial(trial_directory, directions):
    items = []
    next_sentence_id = 1
    for target_direction in directions:
        rows = _read_before_after_csv(
            trial_directory / f"{target_direction}.csv"
        )
        for row in rows:
            items.append(
                EvaluationItem(
                    sentence_id=next_sentence_id,
                    target_direction=target_direction,
                    text_before=row["text_before"],
                    text_after=row["text_after"],
                )
            )
            next_sentence_id += 1

    expected_total = EXPECTED_EXAMPLES_PER_DIRECTION * len(directions)
    if len(items) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} combined examples in {trial_directory}, "
            f"found {len(items)}."
        )
    return items


def _load_classifier():
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is set to CUDA, but CUDA is not available.")

    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            CLASSIFIER_MODEL_ID,
            torch_dtype=dtype,
            local_files_only=LOCAL_FILES_ONLY,
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def _classification_probabilities(model, tokenizer, texts, choices):
    probabilities = []
    for start in range(0, len(texts), CLASSIFICATION_BATCH_SIZE):
        probabilities.extend(
            eval_temp_classification(
                model=model,
                tokenizer=tokenizer,
                text_after=texts[start:start + CLASSIFICATION_BATCH_SIZE],
                choices=list(choices),
            )
        )
    if len(probabilities) != len(texts):
        raise RuntimeError("The classifier returned the wrong number of results.")
    return probabilities


def _steering_metrics(
    items,
    directions,
    model,
    tokenizer,
    source_probability_cache,
):
    after_probabilities = _classification_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=[item.text_after for item in items],
        choices=directions,
    )
    cache_keys = [
        (tuple(directions), item.text_before)
        for item in items
    ]
    missing_keys = list(dict.fromkeys(
        key for key in cache_keys if key not in source_probability_cache
    ))
    if missing_keys:
        missing_probabilities = _classification_probabilities(
            model=model,
            tokenizer=tokenizer,
            texts=[key[1] for key in missing_keys],
            choices=directions,
        )
        source_probability_cache.update(zip(missing_keys, missing_probabilities))
    before_probabilities = [source_probability_cache[key] for key in cache_keys]
    successes = []
    target_probability_gains = []
    for item, before, after in zip(
        items,
        before_probabilities,
        after_probabilities,
    ):
        predicted_index = max(
            range(len(after)),
            key=after.__getitem__,
        )
        predicted_direction = directions[predicted_index]
        successes.append(int(predicted_direction == item.target_direction))
        target_index = directions.index(item.target_direction)
        target_probability_gains.append(
            float(after[target_index]) - float(before[target_index])
        )
    return successes, target_probability_gains


def _judge_scores(items):
    sources = [item.text_before for item in items]
    rewrites = [item.text_after for item in items]
    judge_kwargs = {
        "model": LLM_JUDGE_MODEL,
        "batch_size": LLM_JUDGE_BATCH_SIZE,
        "timeout": LLM_JUDGE_TIMEOUT,
    }
    faithfulness_scores = llmjudge_faithfulness(
        sources,
        rewrites,
        **judge_kwargs,
    )
    structure_scores = llmjudge_retain_structure(
        sources,
        rewrites,
        **judge_kwargs,
    )
    if not (
        len(faithfulness_scores) == len(structure_scores) == len(items)
    ):
        raise RuntimeError("An LLM judge returned the wrong number of scores.")
    return faithfulness_scores, structure_scores


def _write_evaluation(
    output_path,
    items,
    steering_successes,
    target_probability_gains,
    faithfulness_scores,
    structure_scores,
):
    row_count = len(items)
    if not (
        len(steering_successes)
        == len(target_probability_gains)
        == len(faithfulness_scores)
        == len(structure_scores)
        == row_count
    ):
        raise ValueError("All evaluation result lists must have the same length.")

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, item in enumerate(items):
            writer.writerow(
                {
                    "sentence_id": item.sentence_id,
                    "succesfully_steered": steering_successes[index],
                    "target_probability_gain": round(
                        target_probability_gains[index],
                        9,
                    ),
                    "faithfulness_score": faithfulness_scores[index],
                    "structure_retention_score": structure_scores[index],
                }
            )
    temporary_path.replace(output_path)


def _read_existing_judge_scores(path, expected_rows):
    """Reuse paid judge scores from a complete legacy or current output."""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sentence_id",
            "faithfulness_score",
            "structure_retention_score",
        }
        if not required.issubset(reader.fieldnames or []):
            return None
        rows = list(reader)
    if len(rows) != expected_rows:
        return None
    expected_ids = [str(index) for index in range(1, expected_rows + 1)]
    if [row["sentence_id"] for row in rows] != expected_ids:
        return None
    return (
        [int(row["faithfulness_score"]) for row in rows],
        [int(row["structure_retention_score"]) for row in rows],
    )


def _output_is_complete(path, expected_rows):
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            return False
        rows = list(reader)
    return len(rows) == expected_rows


def _validate_all_inputs():
    trials_by_dataset = {}
    total_trials = 0
    total_examples = 0
    for dataset_name, directions in DATASET_DIRECTIONS.items():
        dataset_root, trial_directories = _find_trial_directories(
            dataset_name,
            directions,
        )
        for trial_directory in trial_directories:
            items = _read_trial(trial_directory, directions)
            total_examples += len(items)
        trials_by_dataset[dataset_name] = (
            dataset_root,
            directions,
            trial_directories,
        )
        total_trials += len(trial_directories)
    return trials_by_dataset, total_trials, total_examples



CLASSIFICATION_FIELDNAMES = [
    "sentence_id",
    "succesfully_steered",
    "target_probability_gain",
]


def _write_classification(path, items, successes, gains):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLASSIFICATION_FIELDNAMES)
        writer.writeheader()
        for item, success, gain in zip(items, successes, gains):
            writer.writerow(
                {
                    "sentence_id": item.sentence_id,
                    "succesfully_steered": success,
                    "target_probability_gain": round(gain, 9),
                }
            )
    temporary_path.replace(path)


def _read_classification(path, expected_rows):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CLASSIFICATION_FIELDNAMES:
            return None
        rows = list(reader)
    if len(rows) != expected_rows:
        return None
    expected_ids = [str(index) for index in range(1, expected_rows + 1)]
    if [row["sentence_id"] for row in rows] != expected_ids:
        return None
    return (
        [int(row["succesfully_steered"]) for row in rows],
        [float(row["target_probability_gain"]) for row in rows],
    )

def main():
    args = _parse_args()
    trials_by_dataset, total_trials, total_examples = _validate_all_inputs()
    print(
        f"Validated {total_trials} trials containing "
        f"{total_examples} generated texts."
    )
    if args.validate_only:
        return

    expected_rows = (
        EXPECTED_EXAMPLES_PER_DIRECTION
        * len(next(iter(DATASET_DIRECTIONS.values())))
    )
    classification_jobs = []
    for dataset_name, (dataset_root, directions, trial_directories) in (
        trials_by_dataset.items()
    ):
        for trial_directory in trial_directories:
            output_path = trial_directory / OUTPUT_FILENAME
            checkpoint_path = trial_directory / CLASSIFICATION_FILENAME
            if (
                not args.overwrite
                and _output_is_complete(output_path, expected_rows)
            ):
                continue
            if (
                args.overwrite
                or _read_classification(checkpoint_path, expected_rows) is None
            ):
                classification_jobs.append(
                    (dataset_name, dataset_root, directions, trial_directory)
                )

    if classification_jobs:
        model, tokenizer = _load_classifier()
        source_probability_cache = {}
        with tqdm(
            total=len(classification_jobs),
            desc="Qwen classification",
            unit="trial",
        ) as progress:
            for _, dataset_root, directions, trial_directory in classification_jobs:
                progress.set_postfix_str(
                    str(trial_directory.relative_to(dataset_root)),
                    refresh=True,
                )
                items = _read_trial(trial_directory, directions)
                successes, gains = _steering_metrics(
                    items=items,
                    directions=directions,
                    model=model,
                    tokenizer=tokenizer,
                    source_probability_cache=source_probability_cache,
                )
                checkpoint_path = trial_directory / CLASSIFICATION_FILENAME
                _write_classification(checkpoint_path, items, successes, gains)
                tqdm.write(f"Wrote {checkpoint_path}")
                progress.update()
        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print("All Qwen classification checkpoints are already complete.")

    if args.classification_only:
        print("Classification-only evaluation complete.")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Qwen checkpoints are complete; "
            "set the key and rerun this script to add the two LLM-judge metrics."
        )

    with tqdm(
        total=total_trials,
        desc="LLM-judge evaluation",
        unit="trial",
    ) as progress:
        for _, (dataset_root, directions, trial_directories) in (
            trials_by_dataset.items()
        ):
            for trial_directory in trial_directories:
                output_path = trial_directory / OUTPUT_FILENAME
                progress.set_postfix_str(
                    str(trial_directory.relative_to(dataset_root)),
                    refresh=True,
                )
                if (
                    not args.overwrite
                    and _output_is_complete(output_path, expected_rows)
                ):
                    tqdm.write(f"Skipping complete output: {output_path}")
                    progress.update()
                    continue

                items = _read_trial(trial_directory, directions)
                checkpoint_path = trial_directory / CLASSIFICATION_FILENAME
                classification = _read_classification(
                    checkpoint_path,
                    expected_rows,
                )
                if classification is None:
                    raise RuntimeError(
                        f"Missing or incomplete checkpoint: {checkpoint_path}"
                    )
                successes, gains = classification
                faithfulness_scores, structure_scores = _judge_scores(items)
                _write_evaluation(
                    output_path=output_path,
                    items=items,
                    steering_successes=successes,
                    target_probability_gains=gains,
                    faithfulness_scores=faithfulness_scores,
                    structure_scores=structure_scores,
                )
                tqdm.write(f"Wrote {output_path}")
                progress.update()


if __name__ == "__main__":
    main()
