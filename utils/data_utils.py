import random
import csv
from datasets import load_dataset
from pathlib import Path

def get_imdb(sample_size, seed=42):
    random.seed(seed)

    imdb = load_dataset("imdb")
    train = list(imdb["train"])
    test = imdb["test"]

    random.shuffle(train)

    pos_texts, neg_texts = [], []

    for ex in train:
        if ex["label"] == 1 and len(pos_texts) < sample_size:
            pos_texts.append(ex["text"])
        elif ex["label"] == 0 and len(neg_texts) < sample_size:
            neg_texts.append(ex["text"])

        if len(pos_texts) == sample_size and len(neg_texts) == sample_size:
            break

    test_texts = [ex["text"] for ex in test]
    test_labels = [ex["label"] for ex in test]

    return (pos_texts, neg_texts), (test_texts, test_labels)

def load_timpa_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    dataset_key = dataset_path.as_posix().lower()
    data = {}

    def load_csv(path, class_column=None, fixed_class=None):
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "text" not in reader.fieldnames:
                return
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                if fixed_class is not None:
                    class_name = fixed_class
                elif class_column is not None and row.get(class_column):
                    class_name = row[class_column].strip()
                else:
                    label = (row.get("label") or "").strip()
                    if not label:
                        continue
                    class_name = "positive" if label == "1" else "negative" if label == "0" else label
                data.setdefault(class_name, []).append(text)

    if "cats_dogs" in dataset_key or "cat_dogs" in dataset_key:
        load_csv(dataset_path / "train.csv", class_column="concept")
    elif "imdb" in dataset_key:
        load_csv(dataset_path / "train_pos.csv", fixed_class="positive")
        load_csv(dataset_path / "train_neg.csv", fixed_class="negative")
    else:
        raise NotImplementedError("Decide on a unified format pls")
    return data