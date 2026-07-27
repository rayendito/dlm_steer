import csv
import json
from pathlib import Path

from timpa_eval import (
    elifive_order_mae,
    llmjudge_elifive_order,
    llmjudge_factuality,
    llmjudge_faithfulness,
    llmjudge_retain_structure,
)
from tqdm.auto import tqdm


#### LLM JUDGE
LLM_JUDGE_MODEL = "openai/gpt-4.1-mini"
LLM_JUDGE_BATCH_SIZE = 10
ORDER_SEED = 0

#### INPUTS AND OUTPUT
TEST_ROOT = Path("timpateks_results/elifive_test_csv")
EXPECTED_EXAMPLES = 100
FIVE_YEAR_OLD_FILENAME = "5yo.csv"
HIGH_SCHOOL_FILENAME = "highschool.csv"
PHD_FILENAME = "phd.csv"
OUTPUT_FILENAME = "llmjudge_eval.csv"

FIELDNAMES = [
    "sentence_id",
    "5yo_factuality",
    "highschool_factuality",
    "phd_factuality",
    "5yo_faithfulness",
    "highschool_faithfulness",
    "phd_faithfulness",
    "5yo_retain",
    "highschool_retain",
    "phd_retain",
    "order",
    "order_mae",
]


def read_before_after_csv(path):
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

    if not rows:
        raise ValueError(f"Evaluation input is empty: {path}")
    if any(
        not row["text_before"] or not row["text_after"]
        for row in rows
    ):
        raise ValueError(f"{path} contains an empty before/after text.")
    return {
        "before": [row["text_before"] for row in rows],
        "after": [row["text_after"] for row in rows],
    }


def validate_aligned_inputs(five_year_old, high_school, phd):
    row_counts = {
        len(five_year_old["before"]),
        len(high_school["before"]),
        len(phd["before"]),
    }
    if len(row_counts) != 1:
        raise ValueError(
            "The 5yo, high-school, and PhD CSVs must have the same row count."
        )
    row_count = next(iter(row_counts))
    if row_count != EXPECTED_EXAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_EXAMPLES} test examples, found {row_count}."
        )
    if not (
        five_year_old["before"]
        == high_school["before"]
        == phd["before"]
    ):
        raise ValueError(
            "The source texts must match in the same order across all three CSVs."
        )


def find_experiment_directories():
    if not TEST_ROOT.is_dir():
        raise FileNotFoundError(
            f"ELI5 test directory does not exist: {TEST_ROOT}"
        )

    experiment_directories = sorted(
        path.parent
        for path in TEST_ROOT.glob(f"seed*/*/{FIVE_YEAR_OLD_FILENAME}")
    )
    if not experiment_directories:
        raise FileNotFoundError(
            f"No completed ELI5 experiments were found under {TEST_ROOT}."
        )

    required_filenames = {
        FIVE_YEAR_OLD_FILENAME,
        HIGH_SCHOOL_FILENAME,
        PHD_FILENAME,
    }
    for experiment_directory in experiment_directories:
        missing = [
            filename
            for filename in sorted(required_filenames)
            if not (experiment_directory / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{experiment_directory} is missing required inputs: "
                f"{', '.join(missing)}."
            )
    return experiment_directories


def evaluate_experiment(experiment_directory):
    five_year_old = read_before_after_csv(
        experiment_directory / FIVE_YEAR_OLD_FILENAME
    )
    high_school = read_before_after_csv(
        experiment_directory / HIGH_SCHOOL_FILENAME
    )
    phd = read_before_after_csv(experiment_directory / PHD_FILENAME)
    validate_aligned_inputs(five_year_old, high_school, phd)

    source_texts = five_year_old["before"]
    five_year_old_texts = five_year_old["after"]
    high_school_texts = high_school["after"]
    phd_texts = phd["after"]
    judge_kwargs = {
        "model": LLM_JUDGE_MODEL,
        "batch_size": LLM_JUDGE_BATCH_SIZE,
    }

    five_year_old_factuality = llmjudge_factuality(
        five_year_old_texts,
        **judge_kwargs,
    )
    high_school_factuality = llmjudge_factuality(
        high_school_texts,
        **judge_kwargs,
    )
    phd_factuality = llmjudge_factuality(
        phd_texts,
        **judge_kwargs,
    )

    five_year_old_faithfulness = llmjudge_faithfulness(
        source_texts,
        five_year_old_texts,
        **judge_kwargs,
    )
    high_school_faithfulness = llmjudge_faithfulness(
        source_texts,
        high_school_texts,
        **judge_kwargs,
    )
    phd_faithfulness = llmjudge_faithfulness(
        source_texts,
        phd_texts,
        **judge_kwargs,
    )

    five_year_old_retain = llmjudge_retain_structure(
        source_texts,
        five_year_old_texts,
        **judge_kwargs,
    )
    high_school_retain = llmjudge_retain_structure(
        source_texts,
        high_school_texts,
        **judge_kwargs,
    )
    phd_retain = llmjudge_retain_structure(
        source_texts,
        phd_texts,
        **judge_kwargs,
    )

    orders = llmjudge_elifive_order(
        list(
            zip(
                five_year_old_texts,
                high_school_texts,
                phd_texts,
            )
        ),
        seed=ORDER_SEED,
        **judge_kwargs,
    )
    order_maes = elifive_order_mae(orders)

    output_csv = experiment_directory / OUTPUT_FILENAME
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row_index in range(len(source_texts)):
            writer.writerow(
                {
                    "sentence_id": row_index + 1,
                    "5yo_factuality": five_year_old_factuality[row_index],
                    "highschool_factuality": high_school_factuality[row_index],
                    "phd_factuality": phd_factuality[row_index],
                    "5yo_faithfulness": five_year_old_faithfulness[row_index],
                    "highschool_faithfulness": (
                        high_school_faithfulness[row_index]
                    ),
                    "phd_faithfulness": phd_faithfulness[row_index],
                    "5yo_retain": five_year_old_retain[row_index],
                    "highschool_retain": high_school_retain[row_index],
                    "phd_retain": phd_retain[row_index],
                    "order": json.dumps(orders[row_index]),
                    "order_mae": round(order_maes[row_index], 6),
                }
            )
    return output_csv


def main():
    experiment_directories = find_experiment_directories()
    with tqdm(
        experiment_directories,
        desc="ELI5 LLM-judge evaluation",
        unit="experiment",
    ) as progress:
        for experiment_directory in progress:
            progress.set_postfix_str(
                str(experiment_directory.relative_to(TEST_ROOT)),
                refresh=True,
            )
            output_csv = evaluate_experiment(experiment_directory)
            tqdm.write(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
