import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .sweep_utils import eval_temp_classification


@dataclass(frozen=True)
class SteerSweepEvalConfig:
    dataset_name: str
    target_directions: tuple[str, str]
    input_root: Path
    model_id: str
    device: str = "cuda"
    local_files_only: bool = True
    expected_examples: int = 10
    batch_size: int = 5
    output_filename: str = "classification_eval.csv"


def _validate_config(config):
    if len(set(config.target_directions)) != 2:
        raise ValueError("target_directions must contain two distinct labels.")
    if not isinstance(config.expected_examples, int) or config.expected_examples <= 0:
        raise ValueError("expected_examples must be a positive integer.")
    if not isinstance(config.batch_size, int) or config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not config.output_filename.endswith(".csv"):
        raise ValueError("output_filename must end in '.csv'.")


def _load_classifier(config):
    dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            local_files_only=config.local_files_only,
        )
        .to(config.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def _read_direction_csv(path, expected_direction, expected_examples):
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation input does not exist: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "text_before",
            "text_after",
            "example_id",
            "seed",
            "target_direction",
            "source_direction",
            "masked_token_fraction",
            "layer",
            "alpha",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing columns: {missing}.")
        rows = list(reader)

    if len(rows) != expected_examples:
        raise ValueError(
            f"Expected {expected_examples} rows in {path}, found {len(rows)}."
        )
    if any(
        not row["text_before"].strip() or not row["text_after"].strip()
        for row in rows
    ):
        raise ValueError(f"{path} contains an empty before/after text.")
    if any(
        row["target_direction"] != expected_direction
        for row in rows
    ):
        raise ValueError(
            f"{path} contains a target direction other than "
            f"{expected_direction!r}."
        )
    return rows


def _find_experiment_directories(config):
    if not config.input_root.is_dir():
        raise FileNotFoundError(
            f"{config.dataset_name} sweep directory does not exist: "
            f"{config.input_root}"
        )

    first_direction = config.target_directions[0]
    experiment_directories = sorted(
        path.parent
        for path in config.input_root.glob(
            f"seed*/add_layer*_alpha*/{first_direction}.csv"
        )
    )
    if not experiment_directories:
        raise FileNotFoundError(
            f"No completed {config.dataset_name} steering experiments were "
            f"found under {config.input_root}."
        )

    for experiment_directory in experiment_directories:
        missing = [
            f"{direction}.csv"
            for direction in config.target_directions
            if not (experiment_directory / f"{direction}.csv").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{experiment_directory} is missing required inputs: "
                f"{', '.join(missing)}."
            )
    return experiment_directories


def _classification_probabilities(
    model,
    tokenizer,
    texts,
    choices,
    batch_size,
):
    probabilities = []
    for start in range(0, len(texts), batch_size):
        probabilities.extend(
            eval_temp_classification(
                model=model,
                tokenizer=tokenizer,
                text_after=texts[start:start + batch_size],
                choices=list(choices),
            )
        )
    return probabilities


def _output_fieldnames(config):
    return [
        "sentence_id",
        "target_direction",
        "source_direction",
        "text_before",
        "text_after",
        *(
            f"{direction}_probability"
            for direction in config.target_directions
        ),
        "target_probability",
        "source_probability",
        "predicted_class",
        "correct",
        "masked_token_fraction",
        "layer",
        "alpha",
        "seed",
    ]


def _atomic_write_csv(path, fieldnames, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _evaluate_experiment(
    config,
    experiment_directory,
    model,
    tokenizer,
):
    output_rows = []
    for target_direction in config.target_directions:
        input_rows = _read_direction_csv(
            experiment_directory / f"{target_direction}.csv",
            expected_direction=target_direction,
            expected_examples=config.expected_examples,
        )
        probabilities = _classification_probabilities(
            model=model,
            tokenizer=tokenizer,
            texts=[row["text_after"] for row in input_rows],
            choices=config.target_directions,
            batch_size=config.batch_size,
        )
        target_index = config.target_directions.index(target_direction)
        source_index = 1 - target_index
        source_direction = config.target_directions[source_index]
        if any(
            row["source_direction"] != source_direction
            for row in input_rows
        ):
            raise ValueError(
                f"{experiment_directory / f'{target_direction}.csv'} contains "
                f"a source direction other than {source_direction!r}."
            )

        for input_row, class_probabilities in zip(input_rows, probabilities):
            predicted_index = max(
                range(len(class_probabilities)),
                key=class_probabilities.__getitem__,
            )
            output_row = {
                "sentence_id": input_row["example_id"],
                "target_direction": target_direction,
                "source_direction": source_direction,
                "text_before": input_row["text_before"],
                "text_after": input_row["text_after"],
                "target_probability": class_probabilities[target_index],
                "source_probability": class_probabilities[source_index],
                "predicted_class": config.target_directions[predicted_index],
                "correct": int(predicted_index == target_index),
                "masked_token_fraction": input_row["masked_token_fraction"],
                "layer": input_row["layer"],
                "alpha": input_row["alpha"],
                "seed": input_row["seed"],
            }
            for direction, probability in zip(
                config.target_directions,
                class_probabilities,
            ):
                output_row[f"{direction}_probability"] = probability
            output_rows.append(output_row)

    output_path = experiment_directory / config.output_filename
    _atomic_write_csv(
        output_path,
        _output_fieldnames(config),
        output_rows,
    )
    return output_path


def run_steer_sweep_evaluation(config):
    """Classify every raw steering-sweep generation with one local model."""
    _validate_config(config)
    experiment_directories = _find_experiment_directories(config)
    model, tokenizer = _load_classifier(config)

    with tqdm(
        experiment_directories,
        desc=f"{config.dataset_name} Qwen classification",
        unit="experiment",
    ) as progress:
        for experiment_directory in progress:
            progress.set_postfix_str(
                str(experiment_directory.relative_to(config.input_root)),
                refresh=True,
            )
            output_path = _evaluate_experiment(
                config=config,
                experiment_directory=experiment_directory,
                model=model,
                tokenizer=tokenizer,
            )
            tqdm.write(f"Wrote {output_path}")
