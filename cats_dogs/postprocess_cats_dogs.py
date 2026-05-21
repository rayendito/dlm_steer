from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path


EXPLICIT_ANIMAL_TOKENS = {
    "cat", "cats", "kitten", "kittens", "feline", "felines",
    "dog", "dogs", "puppy", "puppies", "canine", "canines",
}


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def weak_signature(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_weird_sentence(s: str, min_words: int = 6, max_words: int = 45) -> bool:
    if not s:
        return True
    ws = s.split()
    if len(ws) < min_words or len(ws) > max_words:
        return True
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 0.5 * max(1, len(s)):
        return True
    if re.search(r"(.)\1{5,}", s):
        return True
    return False


def has_explicit_animal_leak(s: str, target_concept: str) -> bool:
    toks = re.findall(r"[a-z]+", s.lower())
    if not toks:
        return True
    token_set = set(toks)
    if target_concept not in token_set:
        return True
    # Allow target concept token itself, block class/type names.
    leak_set = EXPLICIT_ANIMAL_TOKENS - {target_concept, target_concept + "s"}
    return any(tok in leak_set for tok in token_set)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "concept", "prompt_kind"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Postprocess raw cats/dogs generations into final dataset.")
    ap.add_argument("--input-jsonl", type=Path, default=Path("benchmarks/cats_dogs/raw_generations.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/cats_dogs"))
    ap.add_argument("--per-class-target", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    seen_exact: dict[str, set[str]] = defaultdict(set)
    seen_weak: dict[str, set[str]] = defaultdict(set)

    raw_rows = 0
    kept_rows = 0
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"Missing input file: {args.input_jsonl}")

    with args.input_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows += 1
            row = json.loads(line)
            concept = str(row["concept"]).strip().lower()
            prompt_kind = str(row.get("prompt_kind", "unknown"))
            for s in row.get("sentences", []):
                s_norm = normalize_text(str(s))
                if is_weird_sentence(s_norm):
                    continue
                if has_explicit_animal_leak(s_norm, target_concept=concept):
                    continue
                sig_exact = s_norm.lower()
                sig_weak = weak_signature(s_norm)
                if sig_exact in seen_exact[concept] or sig_weak in seen_weak[concept]:
                    continue
                seen_exact[concept].add(sig_exact)
                seen_weak[concept].add(sig_weak)
                pools[concept].append(
                    {
                        "text": s_norm,
                        "label": 0 if concept == "cat" else 1,
                        "concept": concept,
                        "prompt_kind": prompt_kind,
                    }
                )
                kept_rows += 1

    final_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for concept in ("cat", "dog"):
        items = pools[concept]
        rng.shuffle(items)
        if len(items) < args.per_class_target:
            raise RuntimeError(
                f"Not enough filtered rows for {concept}: got {len(items)}, need {args.per_class_target}. "
                "Increase oversampling and regenerate."
            )
        items = items[: args.per_class_target]
        for i, row in enumerate(items):
            row["id"] = f"{concept}-{i:04d}"
        final_rows.extend(items)
        summary[concept] = {
            "filtered_pool_size": len(pools[concept]),
            "selected_final": len(items),
        }

    rng.shuffle(final_rows)
    n_total = len(final_rows)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_rows = final_rows[:n_train]
    val_rows = final_rows[n_train:n_train + n_val]
    test_rows = final_rows[n_train + n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all.csv", final_rows)
    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "val.csv", val_rows)
    write_csv(args.output_dir / "test.csv", test_rows)

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "raw_rows": raw_rows,
        "kept_candidate_rows": kept_rows,
        "summary_by_concept": summary,
        "splits": {"train": n_train, "val": n_val, "test": n_test},
        "per_class_target": args.per_class_target,
        "seed": args.seed,
    }
    with (args.output_dir / "postprocess_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
