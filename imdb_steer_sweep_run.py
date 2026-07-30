from pathlib import Path

from timpa_experimental import (
    SteerVectorSweepConfig,
    run_steer_vector_sweep,
)


#### MODELING
DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
LOCAL_FILES_ONLY = True

#### VECTOR SELECTION
# None searches every transformer layer.
LAYER_CANDIDATES = None
ADD_ALPHAS = (100.0, 300.0, 600.0, 900.0, 1200.0)
RANDOM_MASK_PROBABILITY = 0.5

#### GENERATION
RANDOM_SEEDS = (42,)
GENERATION_BATCH_SIZE = 5
REFILL_STEPS = 32
SAMPLING_TEMPERATURE = 0.1
REFILL_STRATEGY = "low_confidence"

#### OUTPUTS
CSV_ROOT = Path("timpateks_results/imdb_steer_sweep_csv")
HTML_ROOT = Path("timpateks_results/imdb_steer_sweep_html")


def main():
    run_steer_vector_sweep(
        SteerVectorSweepConfig(
            dataset_name="imdb",
            target_directions=("positive", "negative"),
            output_csv_root=CSV_ROOT,
            output_html_root=HTML_ROOT,
            model_id=MODEL_ID,
            device=DEVICE,
            local_files_only=LOCAL_FILES_ONLY,
            split="train",
            layer_candidates=LAYER_CANDIDATES,
            add_alphas=ADD_ALPHAS,
            random_seeds=RANDOM_SEEDS,
            generation_batch_size=GENERATION_BATCH_SIZE,
            refill_steps=REFILL_STEPS,
            sampling_temperature=SAMPLING_TEMPERATURE,
            random_mask_probability=RANDOM_MASK_PROBABILITY,
            refill_strategy=REFILL_STRATEGY,
        )
    )


if __name__ == "__main__":
    main()
