#!/usr/bin/env python3
from __future__ import annotations

import subprocess


N_SEEDS = {
    0: [41],
    5: [41, 42, 43],
    10: [41, 42, 43],
    15: [41, 42, 43],
    20: [41, 42, 43],
    30: [41, 42, 43],
    40: [41, 42, 43],
    50: [41, 42, 43],
}
QUEUES = ["research2", "research"]
PYTHON = "/share03/afz225/miniconda3/bin/python"


def main() -> None:
    submitted = []
    i = 0
    for n, seeds in N_SEEDS.items():
        for seed in seeds:
            queue = QUEUES[i % len(QUEUES)]
            i += 1
            name = f"cdogs_n{n}_s{seed}"
            body = (
                'cd "$PBS_O_WORKDIR"; '
                f"{PYTHON} extract_vectors/run_vector_count_ablation.py "
                "--concept-pair cat-dog --pair-order cat,dog "
                "--vector-source-split val "
                f"--n-values {n} --seeds {seed} --skip-existing"
            )
            cmd = [
                "qsub",
                "-q",
                queue,
                "-N",
                name,
                "-l",
                "select=1:ngpus=1:ncpus=8",
                "-l",
                "walltime=24:00:00",
                "--",
                "/bin/bash",
                "-lc",
                body,
            ]
            job_id = subprocess.check_output(cmd, text=True).strip()
            submitted.append((job_id, queue, name))
            print(f"{job_id}\t{queue}\t{name}", flush=True)
    print(f"submitted={len(submitted)}", flush=True)


if __name__ == "__main__":
    main()
