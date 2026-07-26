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


#### LLM JUDGE
LLM_JUDGE_MODEL = "openai/gpt-4.1-mini"
LLM_JUDGE_BATCH_SIZE = 10
ORDER_SEED = 0

#### INPUTS AND OUTPUT
FIVE_YEAR_OLD_CSV = Path("timpaprobs_elifive_to_5yo_random.csv")
HIGH_SCHOOL_CSV = Path("timpaprobs_elifive_to_highschool_random.csv")
PHD_CSV = Path("timpaprobs_elifive_to_phd_random.csv")
OUTPUT_CSV = Path("timpaprobs_elifive_random_llmjudge_eval.csv")

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
    if not (
        five_year_old["before"]
        == high_school["before"]
        == phd["before"]
    ):
        raise ValueError(
            "The source texts must match in the same order across all three CSVs."
        )


def main():
    five_year_old = read_before_after_csv(FIVE_YEAR_OLD_CSV)
    high_school = read_before_after_csv(HIGH_SCHOOL_CSV)
    phd = read_before_after_csv(PHD_CSV)
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

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
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

    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
