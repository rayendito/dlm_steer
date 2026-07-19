import csv
from pathlib import Path


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


def timpa_load_data(dataset_name):
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


__all__ = ["timpa_load_data"]
