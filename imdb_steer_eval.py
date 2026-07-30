from pathlib import Path

from timpa_eval import SteerSweepEvalConfig, run_steer_sweep_evaluation


#### MODELING
DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
LOCAL_FILES_ONLY = True
CLASSIFICATION_BATCH_SIZE = 5

#### INPUTS AND OUTPUT
INPUT_ROOT = Path("timpateks_results/imdb_steer_sweep_csv")
EXPECTED_EXAMPLES = 10
OUTPUT_FILENAME = "classification_eval.csv"


def main():
    run_steer_sweep_evaluation(
        SteerSweepEvalConfig(
            dataset_name="imdb",
            target_directions=("positive", "negative"),
            input_root=INPUT_ROOT,
            model_id=MODEL_ID,
            device=DEVICE,
            local_files_only=LOCAL_FILES_ONLY,
            expected_examples=EXPECTED_EXAMPLES,
            batch_size=CLASSIFICATION_BATCH_SIZE,
            output_filename=OUTPUT_FILENAME,
        )
    )


if __name__ == "__main__":
    main()

