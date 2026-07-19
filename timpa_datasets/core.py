import csv
from pathlib import Path

from timpa_steer_vectors import extract_steer_vectors, extract_steer_vectors_add


DATASET_ROOT = Path(__file__).resolve().parent


def _read_rows(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = set(required_columns) - fieldnames
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing}.")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            cleaned = {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            for column in required_columns:
                if not cleaned.get(column):
                    raise ValueError(
                        f"{path}:{line_number} has an empty {column!r} value."
                    )
            rows.append(cleaned)

    if not rows:
        raise ValueError(f"Dataset split is empty: {path}")
    return rows


def _load_split(dataset_name, path):
    paired = dataset_name in {"catdog", "imdb"}
    required_columns = ["text1", "text2"] if paired else ["text1"]
    rows = _read_rows(path, required_columns)

    if dataset_name == "catdog":
        return {
            "cat": [row["text1"] for row in rows],
            "dog": [row["text2"] for row in rows],
        }
    if dataset_name == "imdb":
        return {
            "positive": [row["text1"] for row in rows],
            "negative": [row["text2"] for row in rows],
        }
    return {"text": [row["text1"] for row in rows]}


def timpa_load_rows(dataset_name):
    """Load the train and test splits for one TIMPA dataset."""
    dataset_names = {"catdog", "imdb", "elifive", "dolly_sample"}
    if dataset_name not in dataset_names:
        available = ", ".join(sorted(dataset_names))
        raise ValueError(
            f"Unknown TIMPA dataset {dataset_name!r}. Available names: {available}."
        )

    dataset_path = DATASET_ROOT / dataset_name
    train_path = dataset_path / "train.csv"
    if not train_path.is_file():
        train_path = dataset_path / "val.csv"

    return {
        "dataset": {
            "train": _load_split(dataset_name, train_path),
            "test": _load_split(dataset_name, dataset_path / "test.csv"),
        }
    }


def _ar_steer_prompts(dataset_name):
    prompts = {
        "imdb": {
            "positive": "Rewrite this sentence as a positive movie review.",
            "negative": "Rewrite this sentence as a negative movie review.",
        },
        "catdog": {
            "dog": "Rewrite this sentence so that it is about dogs.",
            "cat": "Rewrite this sentence so that it is about cats.",
        },
        "elifive": {
            "5yo": "Explain this as if you were speaking to a five-year-old.",
            "highschool": (
                "Explain this as if you were speaking to a high school student "
                "who is learning about the topic."
            ),
            "phd": (
                "Explain this as if you were speaking to someone with a PhD in "
                "the field."
            ),
        },
        "dolly_sample": {
            "pirate": "Rewrite this in the voice of a pirate.",
            "mean": "Rewrite this in a mean-spirited tone.",
            "flirty": "Rewrite this in a flirtatious tone.",
        },
    }
    return prompts[dataset_name]


def _probabilistic_steer_prompts(dataset_name):
    prompts = {
        "imdb": {
            "positive": (
                "You are a generous movie critic who always gives positive reviews."
            ),
            "negative": (
                "You are a harsh movie critic who always gives negative reviews."
            ),
        },
        "catdog": {
            "dog": "You are an assistant who writes exclusively about dogs.",
            "cat": "You are an assistant who writes exclusively about cats.",
        },
        "elifive": {
            "base": "You explain the topic to a subject-matter expert.",
            "5yo": (
                "You explain the topic to a five-year-old who has not yet "
                "started school."
            ),
            "highschool": (
                "You explain the topic to a high school student who is learning "
                "about it in school."
            ),
            "phd": "You explain the topic to someone with a PhD in the field.",
        },
        "dolly_sample": {
            "base": "You are an assistant who responds in a neutral tone.",
            "pirate": "You are an assistant who responds in the voice of a pirate.",
            "mean": "You are an assistant who responds in a mean tone.",
            "flirty": "You are an assistant who responds in a flirtatious tone.",
        },
    }
    return prompts[dataset_name]


def timpa_load_data_and_steer_artefacts(
    dataset_name,
    split,
    timpa_method,
    model=None,
    tokenizer=None,
    steer_direction=None,
    steer_method=None,
    source_layer=23,
    token_position=-4,
    steer_layers=None,
):
    dataset = timpa_load_rows(dataset_name)["dataset"][split]
    vector = None
    steerprompts = None
    if steer_layers is None:
        raise ValueError("steer_layers must be provided.")
    steer_layers = tuple(steer_layers)

    if timpa_method == "timpa_ar":
        steerprompts = _ar_steer_prompts(dataset_name)
    elif timpa_method == "timpa_probabilistic":
        steerprompts = _probabilistic_steer_prompts(dataset_name)
    elif timpa_method in {"timpa_steer", "timpa_hybrid"}:
        if model is None or tokenizer is None:
            raise ValueError(
                "model and tokenizer are required when the TIMPA method uses "
                "steering vectors."
            )
        if steer_direction not in dataset:
            available = ", ".join(dataset)
            raise ValueError(
                f"Unknown steer direction {steer_direction!r}. "
                f"Available concepts: {available}."
            )
        contrast_directions = [
            concept for concept in dataset if concept != steer_direction
        ]
        if len(contrast_directions) != 1:
            raise ValueError(
                "Steering-vector extraction requires exactly two dataset concepts."
            )
        contrast_direction = contrast_directions[0]
        target_corpus = [steer_direction]
        contrast_corpus = [contrast_direction]

        if timpa_method == "timpa_hybrid" and steer_method != "add":
            raise ValueError("timpa_hybrid requires steer_method='add'.")
        if steer_method == "projection":
            vector = extract_steer_vectors(
                model=model,
                tokenizer=tokenizer,
                corpus1=target_corpus,
                corpus2=contrast_corpus,
                source_layer=source_layer,
                token_position=token_position,
            )
        elif steer_method == "add":
            all_vectors = extract_steer_vectors_add(
                model=model,
                tokenizer=tokenizer,
                corpus1=target_corpus,
                corpus2=contrast_corpus,
            )
            missing_layers = [
                layer for layer in steer_layers if layer not in all_vectors
            ]
            if missing_layers:
                raise ValueError(
                    "The model did not return additive steering vectors for layers "
                    f"{missing_layers}."
                )
            vector = {layer: all_vectors[layer] for layer in steer_layers}
        else:
            raise ValueError("steer_method must be 'projection' or 'add'.")

        if timpa_method == "timpa_hybrid":
            steerprompts = _probabilistic_steer_prompts(dataset_name)
    else:
        raise ValueError(f"Unknown TIMPA method {timpa_method!r}.")

    if timpa_method == "timpa_hybrid":
        steer_artifact = {
            "vector": vector,
            "steerprompts": steerprompts,
        }
    elif vector is not None:
        steer_artifact = vector
    else:
        steer_artifact = steerprompts

    return dataset, steer_artifact
