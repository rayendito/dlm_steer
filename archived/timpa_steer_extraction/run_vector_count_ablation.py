#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


N_VALUES = [0, 5, 10, 15, 20, 30, 40, 50]
SEEDS = [41, 42, 43]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Ablate number of sentences used to construct steering vectors with 3 random seeds, "
            "then run both positive and negative sweeps."
        )
    )
    ap.add_argument(
        "--concept-pair",
        type=str,
        default="cat-dog",
        choices=["sentiment", "cat-dog"],
        help="Concept pair for steer vector extraction.",
    )
    ap.add_argument(
        "--pair-order",
        type=str,
        default="cat,dog",
        help="For cat-dog only: positive,negative order (cat,dog or dog,cat).",
    )
    ap.add_argument(
        "--vector-source-split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Split used for vector construction (requested flipped setting: val).",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root.",
    )
    ap.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Only extract vectors; skip positive/negative sweeps.",
    )
    ap.add_argument(
        "--n-values",
        type=str,
        default=",".join(str(x) for x in N_VALUES),
        help="Comma-separated vector-count values to run.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in SEEDS),
        help="Comma-separated seeds to run. Ignored for n=0 except the first value.",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default="positive,negative",
        help="Comma-separated steering directions to sweep.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sweeps whose eval_scores.json already exists.",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    py = [sys.executable]
    n_values = parse_int_list(args.n_values)
    seeds = parse_int_list(args.seeds)
    directions = parse_str_list(args.directions)
    for direction in directions:
        if direction not in {"positive", "negative"}:
            raise ValueError(f"Invalid direction: {direction}")

    for n in n_values:
        run_seeds = seeds[:1] if n == 0 else seeds
        for seed in run_seeds:
            extract_cmd = py + [
                "extract_vectors/extract_steer_vectors.py",
                "--concept-pair", args.concept_pair,
                "--num-samples", str(n),
                "--sample-seed", str(seed),
                "--source-split", args.vector_source_split,
            ]
            if args.concept_pair == "cat-dog":
                extract_cmd += ["--pair-order", args.pair_order]
            run(extract_cmd, repo)

            if args.skip_sweep:
                continue

            if args.concept_pair == "sentiment":
                tag = "sentiment_n0" if n == 0 else f"sentiment_{args.vector_source_split}_n{n}_s{seed}"
            else:
                pos_c, neg_c = [x.strip().lower() for x in args.pair_order.split(",")]
                if n == 0:
                    tag = f"catdog_{pos_c}_minus_{neg_c}_n0"
                else:
                    tag = (
                        f"catdog_{pos_c}_minus_{neg_c}_"
                        f"{args.vector_source_split}_n{n}_s{seed}"
                    )
            vectors_path = Path("steer_vectors") / f"diffusion-val-{tag}.pt"

            for direction in directions:
                out_dir = Path("extract_vectors") / f"results_{direction[:3]}_{tag}"
                if args.skip_existing and (repo / out_dir / "eval_scores.json").is_file():
                    print(f"Skipping existing {out_dir / 'eval_scores.json'}", flush=True)
                    continue
                run(
                    py + [
                        "extract_vectors/resteer_val_sweep_eval.py",
                        "--direction", direction,
                        "--vectors", str(vectors_path),
                    ],
                    repo,
                )

    print("Done vector-count ablation.")


if __name__ == "__main__":
    main()
