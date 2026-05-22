"""Download stanfordnlp/imdb splits and write balanced CSV benchmarks."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from datasets import load_dataset


def sample_label(
    split_rows: list[dict],
    label: int,
    n: int,
    rng: random.Random,
) -> list[dict]:
    pool = [r for r in split_rows if r["label"] == label]
    if len(pool) < n:
        name = "pos" if label == 1 else "neg"
        raise ValueError(
            f"Not enough {name} rows: need {n}, have {len(pool)}"
        )
    return rng.sample(pool, n)


def rows_to_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label"])
        w.writeheader()
        for r in rows:
            w.writerow({"text": r["text"], "label": r["label"]})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample balanced IMDB CSVs from stanfordnlp/imdb train/test splits."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../benchmarks/imdb"),
        help="Directory for train_pos, train_neg, val_pos, val_neg CSVs",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds = load_dataset("stanfordnlp/imdb")

    test_rows = list(ds["test"])
    train_rows = list(ds["train"])

    val_pos = sample_label(test_rows, label=1, n=2000, rng=rng)
    val_neg = sample_label(test_rows, label=0, n=2000, rng=rng)
    train_pos = sample_label(train_rows, label=1, n=2000, rng=rng)
    train_neg = sample_label(train_rows, label=0, n=2000, rng=rng)

    paths = {
        "train_pos": args.out_dir / "train_pos.csv",
        "train_neg": args.out_dir / "train_neg.csv",
        "val_pos": args.out_dir / "val_pos.csv",
        "val_neg": args.out_dir / "val_neg.csv",
    }
    rows_to_csv(train_pos, paths["train_pos"])
    rows_to_csv(train_neg, paths["train_neg"])
    rows_to_csv(val_pos, paths["val_pos"])
    rows_to_csv(val_neg, paths["val_neg"])

    print(f"Wrote {len(train_pos)} rows -> {paths['train_pos']}")
    print(f"Wrote {len(train_neg)} rows -> {paths['train_neg']}")
    print(f"Wrote {len(val_pos)} rows -> {paths['val_pos']}")
    print(f"Wrote {len(val_neg)} rows -> {paths['val_neg']}")


if __name__ == "__main__":
    main()
