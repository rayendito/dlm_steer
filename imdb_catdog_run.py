import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from timpa_datasets import timpa_load_data_and_steer_artefacts
from timpa_experimental import visualize_timpa_probabilistic
from timpateks import helpers
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


#### MODELING
DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
IDENTIFIER_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
LOCAL_FILES_ONLY = False

#### DATASETS
DATASET_DIRECTIONS = {
    "imdb": ("positive", "negative"),
    "catdog": ("cat", "dog"),
}
SPLIT = "test"

#### FULL TEST
SELECTED_HYPERPARAMETERS = {
    "imdb": {"temperature": 0.25, "margin": 0.1},
    "catdog": {"temperature": 1.0, "margin": 0.5},
}
NEGATIVE_DELTA_TEMPERATURE = 0.5
NEGATIVE_DELTA_MARGIN = 0.0
RANDOM_MASK_PROBABILITY = 0.5
RANDOM_SEEDS = (42, 43, 44)
EXPECTED_EXAMPLES_PER_DIRECTION = 100

#### GENERATION
GENERATION_BATCH_SIZE = 5
REFILL_STEPS = 32
SAMPLING_TEMPERATURE = 0.1
REFILL_STRATEGY = "low_confidence"

#### OUTPUTS
RESULTS_ROOT = Path("timpateks_results")


def _float_description(value):
    return f"{value:g}"


def _seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _test_configurations(dataset_name):
    selected = SELECTED_HYPERPARAMETERS[dataset_name]
    return [
        {"description": "random", "detection_strategy": "random",
         "mask_selection": "sample", "temperature": 1.0, "margin": 0.0},
        {"description": "negative_delta", "detection_strategy": "model",
         "mask_selection": "negative_delta",
         "temperature": NEGATIVE_DELTA_TEMPERATURE,
         "margin": NEGATIVE_DELTA_MARGIN},
        {"description": (
             f"selected_temp{_float_description(selected['temperature'])}_"
             f"margin{_float_description(selected['margin'])}"
         ), "detection_strategy": "model", "mask_selection": "sample",
         "temperature": selected["temperature"], "margin": selected["margin"]},
    ]


def _load_models():
    config = LLaDAConfig.from_pretrained(
        MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = (
        LLaDAModelLM.from_pretrained(
            MODEL_ID,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=LOCAL_FILES_ONLY,
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    tokenizer.padding_side = "left"

    identifier_model = (
        AutoModelForCausalLM.from_pretrained(
            IDENTIFIER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=LOCAL_FILES_ONLY,
        )
        .to(DEVICE)
        .eval()
    )
    identifier_tokenizer = AutoTokenizer.from_pretrained(
        IDENTIFIER_MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    identifier_tokenizer.padding_side = "left"
    return model, tokenizer, identifier_model, identifier_tokenizer


def _opposite_direction(directions, target_direction):
    return next(
        direction for direction in directions if direction != target_direction
    )


def _prepare_jobs(tokenizer, identifier_model, identifier_tokenizer):
    jobs = {}
    total_directions = sum(len(value) for value in DATASET_DIRECTIONS.values())
    with tqdm(
        total=total_directions,
        desc="Caching AR-DLM scores",
        unit="direction",
    ) as progress:
        for dataset_name, directions in DATASET_DIRECTIONS.items():
            dataset, prompts = timpa_load_data_and_steer_artefacts(
                dataset_name,
                SPLIT,
                "timpa_probabilistic",
            )
            jobs[dataset_name] = {}
            for target_direction in directions:
                source_direction = _opposite_direction(
                    directions,
                    target_direction,
                )
                texts = dataset[source_direction]
                if len(texts) != EXPECTED_EXAMPLES_PER_DIRECTION:
                    raise ValueError(
                        f"{dataset_name}/{source_direction} has {len(texts)} "
                        f"examples; expected {EXPECTED_EXAMPLES_PER_DIRECTION}."
                    )
                steer_prompts = [prompts[target_direction]] * len(texts)
                base_prompts = [prompts[source_direction]] * len(texts)
                scores = helpers._probabilistic_token_scores(
                    tokenizer=tokenizer,
                    identifier_model=identifier_model,
                    identifier_tokenizer=identifier_tokenizer,
                    steer=steer_prompts,
                    texts=texts,
                    base_assistant_prompt=base_prompts,
                    use_chat_template=True,
                )
                jobs[dataset_name][target_direction] = {
                    "source_direction": source_direction,
                    "texts": texts,
                    "steer_prompts": steer_prompts,
                    "base_prompts": base_prompts,
                    "scores": scores,
                }
                progress.set_postfix(
                    dataset=dataset_name,
                    target=target_direction,
                    refresh=True,
                )
                progress.update()
    return jobs


def _run_configuration(
    configuration,
    job,
    model,
    tokenizer,
    identifier_metadata,
    random_seed,
):
    texts = job["texts"]
    _seed_everything(random_seed)
    generator = torch.Generator(device=DEVICE).manual_seed(random_seed)

    if configuration["detection_strategy"] == "random":
        tokenized_text, masking_probs, masked_positions = (
            helpers._random_token_detection(
                tokenizer=tokenizer,
                texts=texts,
                probability=RANDOM_MASK_PROBABILITY,
                device=helpers._model_device(model),
                generator=generator,
            )
        )
        result_identifier_model = None
    else:
        tokenized_text, aligned_word_log_deltas = job["scores"]
        tokenized_text, masking_probs, masked_positions = (
            helpers._probabilistic_token_detection_from_scores(
                tokenizer=tokenizer,
                texts=texts,
                tokenized_text=tokenized_text,
                aligned_word_log_deltas=aligned_word_log_deltas,
                temperature=configuration["temperature"],
                margin=configuration["margin"],
                mask_selection=configuration["mask_selection"],
                generator=generator,
            )
        )
        result_identifier_model = identifier_metadata

    attention_mask = tokenized_text.get("attention_mask")
    regenerated_texts = []
    batch_starts = range(0, len(texts), GENERATION_BATCH_SIZE)
    for start in tqdm(
        batch_starts,
        total=(len(texts) + GENERATION_BATCH_SIZE - 1)
        // GENERATION_BATCH_SIZE,
        desc="LLaDA refill",
        unit="batch",
        leave=False,
    ):
        end = min(start + GENERATION_BATCH_SIZE, len(texts))
        batch_attention_mask = (
            attention_mask[start:end]
            if attention_mask is not None
            else None
        )
        regenerated_texts.extend(
            helpers.regenerate_masked_text(
                model=model,
                tokenizer=tokenizer,
                steer=job["steer_prompts"][start:end],
                text=texts[start:end],
                masked_positions=masked_positions[start:end],
                response_attention_mask=batch_attention_mask,
                use_chat_template=True,
                refill_steps=REFILL_STEPS,
                sampling_temperature=SAMPLING_TEMPERATURE,
                refill_strategy=REFILL_STRATEGY,
            )
        )

    helpers._attach_probabilistic_result_context(
        tokenized_text=tokenized_text,
        texts=texts,
        steer_prompts=job["steer_prompts"],
        base_prompts=job["base_prompts"],
        tokenizer=tokenizer,
        model=model,
        identifier_model=result_identifier_model,
        use_chat_template=True,
        temperature=configuration["temperature"],
        margin=configuration["margin"],
        refill_steps=REFILL_STEPS,
        detection_strategy=configuration["detection_strategy"],
        mask_selection=configuration["mask_selection"],
    )
    return tokenized_text, masking_probs, masked_positions, regenerated_texts


def _write_before_after_csv(output_file, texts, regenerated_texts):
    if len(texts) != len(regenerated_texts):
        raise ValueError("texts and regenerated_texts must have the same length.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text_before", "text_after"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "text_before": before,
                "text_after": after,
            }
            for before, after in zip(texts, regenerated_texts)
        )
    temporary_file.replace(output_file)


def _output_is_complete(csv_file, html_file):
    if not csv_file.is_file() or not html_file.is_file():
        return False
    try:
        with csv_file.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    return (
        len(rows) == EXPECTED_EXAMPLES_PER_DIRECTION
        and all(row.get("text_before") and row.get("text_after") for row in rows)
    )


def main():
    model, tokenizer, identifier_model, identifier_tokenizer = _load_models()
    jobs = _prepare_jobs(
        tokenizer,
        identifier_model,
        identifier_tokenizer,
    )

    identifier_metadata = SimpleNamespace(
        config=SimpleNamespace(name_or_path=IDENTIFIER_MODEL_ID)
    )
    del identifier_model, identifier_tokenizer
    torch.cuda.empty_cache()

    total_runs = len(RANDOM_SEEDS) * 3 * sum(
        len(value) for value in DATASET_DIRECTIONS.values()
    )
    with tqdm(total=total_runs, desc="IMDB/CatDog full test", unit="run") as progress:
        for dataset_name, directions in DATASET_DIRECTIONS.items():
            configurations = _test_configurations(dataset_name)
            csv_root = RESULTS_ROOT / f"{dataset_name}_test_csv"
            html_root = RESULTS_ROOT / f"{dataset_name}_test_html"
            for random_seed in RANDOM_SEEDS:
                seed_directory = f"seed{random_seed}"
                for configuration in configurations:
                    description = configuration["description"]
                    csv_directory = csv_root / seed_directory / description
                    html_directory = html_root / seed_directory / description
                    csv_directory.mkdir(parents=True, exist_ok=True)
                    html_directory.mkdir(parents=True, exist_ok=True)

                    for target_direction in directions:
                        progress.set_postfix(
                            dataset=dataset_name,
                            seed=random_seed,
                            config=description,
                            target=target_direction,
                            refresh=True,
                        )
                        output_csv = csv_directory / f"{target_direction}.csv"
                        output_html = html_directory / f"{target_direction}.html"
                        if _output_is_complete(output_csv, output_html):
                            progress.update()
                            continue

                        job = jobs[dataset_name][target_direction]
                        (
                            tokenized_text,
                            masking_probs,
                            masked_positions,
                            regenerated_texts,
                        ) = _run_configuration(
                            configuration=configuration,
                            job=job,
                            model=model,
                            tokenizer=tokenizer,
                            identifier_metadata=identifier_metadata,
                            random_seed=random_seed,
                        )

                        visualize_timpa_probabilistic(
                            tokenized_text,
                            masking_probs,
                            masked_positions,
                            regenerated_texts,
                            output_file=output_html,
                        )
                        _write_before_after_csv(
                            output_csv,
                            job["texts"],
                            regenerated_texts,
                        )
                        progress.update()

            print(f"Wrote {dataset_name} CSV full-test results to {csv_root}")
            print(f"Wrote {dataset_name} HTML full-test results to {html_root}")


if __name__ == "__main__":
    main()
