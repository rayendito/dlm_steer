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
LOCAL_FILES_ONLY = True

#### TEST RUN
SELECTED_TEMPERATURE = 0.5
SELECTED_MARGIN = 0.1
NEGATIVE_DELTA_TEMPERATURE = 0.5
NEGATIVE_DELTA_MARGIN = 0.0
RANDOM_MASK_PROBABILITY = 0.5
RANDOM_SEEDS = (42, 43, 44)
GENERATION_BATCH_SIZE = 10
ELIFIVE_REFILL_STEPS = 32
SAMPLING_TEMPERATURE = 0.1
REFILL_STRATEGY = "low_confidence"

#### OUTPUTS
CSV_ROOT = Path("timpateks_results/elifive_test_csv")
HTML_ROOT = Path("timpateks_results/elifive_test_html")


def write_before_after_csv(output_file, text_before, text_after):
    if len(text_before) != len(text_after):
        raise ValueError(
            "text_before and text_after must have the same length."
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
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
            for before, after in zip(text_before, text_after)
        )
    return output_file


def _float_description(value):
    return f"{value:g}"


def _seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _test_configurations():
    return [
        {
            "description": "random",
            "detection_strategy": "random",
            "mask_selection": "sample",
            "temperature": 1.0,
            "margin": 0.0,
        },
        {
            "description": "negative_delta",
            "detection_strategy": "model",
            "mask_selection": "negative_delta",
            "temperature": NEGATIVE_DELTA_TEMPERATURE,
            "margin": NEGATIVE_DELTA_MARGIN,
        },
        {
            "description": (
                f"temp{_float_description(SELECTED_TEMPERATURE)}_"
                f"margin{_float_description(SELECTED_MARGIN)}"
            ),
            "detection_strategy": "model",
            "mask_selection": "sample",
            "temperature": SELECTED_TEMPERATURE,
            "margin": SELECTED_MARGIN,
        },
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


def _cache_detection_scores(
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    texts,
    base_prompts,
    target_prompts,
):
    cached_scores = {}
    for target_name, steer_prompts in tqdm(
        target_prompts.items(),
        total=len(target_prompts),
        desc="Caching AR-DLM scores",
        unit="target",
    ):
        cached_scores[target_name] = helpers._probabilistic_token_scores(
            tokenizer=tokenizer,
            identifier_model=identifier_model,
            identifier_tokenizer=identifier_tokenizer,
            steer=steer_prompts,
            texts=texts,
            base_assistant_prompt=base_prompts,
            use_chat_template=True,
        )
    return cached_scores


def _run_configuration(
    configuration,
    target_name,
    steer_prompts,
    texts,
    base_prompts,
    model,
    tokenizer,
    identifier_model,
    cached_scores,
    random_seed,
):
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
        tokenized_text, aligned_word_log_deltas = cached_scores[target_name]
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
        result_identifier_model = identifier_model

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
                steer=steer_prompts[start:end],
                text=texts[start:end],
                masked_positions=masked_positions[start:end],
                response_attention_mask=batch_attention_mask,
                use_chat_template=True,
                refill_steps=ELIFIVE_REFILL_STEPS,
                sampling_temperature=SAMPLING_TEMPERATURE,
                refill_strategy=REFILL_STRATEGY,
            )
        )
    helpers._attach_probabilistic_result_context(
        tokenized_text=tokenized_text,
        texts=texts,
        steer_prompts=steer_prompts,
        base_prompts=base_prompts,
        tokenizer=tokenizer,
        model=model,
        identifier_model=result_identifier_model,
        use_chat_template=True,
        temperature=configuration["temperature"],
        margin=configuration["margin"],
        refill_steps=ELIFIVE_REFILL_STEPS,
        detection_strategy=configuration["detection_strategy"],
        mask_selection=configuration["mask_selection"],
    )
    return (
        tokenized_text,
        masking_probs,
        masked_positions,
        regenerated_texts,
    )


def main():
    model, tokenizer, identifier_model, identifier_tokenizer = _load_models()

    elifive_data, elifive_artifact = timpa_load_data_and_steer_artefacts(
        "elifive",
        "test",
        "timpa_probabilistic",
    )
    texts = elifive_data["text"]
    if len(texts) != 100:
        raise RuntimeError(
            f"Expected the full 100-example ELI5 test set, found {len(texts)}."
        )
    base_prompts = [elifive_artifact["base"]] * len(texts)
    target_prompts = {
        "5yo": [elifive_artifact["5yo"]] * len(texts),
        "highschool": [elifive_artifact["highschool"]] * len(texts),
        "phd": [elifive_artifact["phd"]] * len(texts),
    }
    cached_scores = _cache_detection_scores(
        tokenizer=tokenizer,
        identifier_model=identifier_model,
        identifier_tokenizer=identifier_tokenizer,
        texts=texts,
        base_prompts=base_prompts,
        target_prompts=target_prompts,
    )
    identifier_metadata = SimpleNamespace(
        config=SimpleNamespace(name_or_path=IDENTIFIER_MODEL_ID)
    )
    del identifier_model, identifier_tokenizer
    torch.cuda.empty_cache()

    configurations = _test_configurations()
    total_runs = len(RANDOM_SEEDS) * len(configurations) * len(target_prompts)
    with tqdm(total=total_runs, desc="ELI5 test", unit="run") as progress:
        for random_seed in RANDOM_SEEDS:
            seed_directory = f"seed{random_seed}"
            for configuration in configurations:
                description = configuration["description"]
                csv_directory = CSV_ROOT / seed_directory / description
                html_directory = HTML_ROOT / seed_directory / description
                csv_directory.mkdir(parents=True, exist_ok=True)
                html_directory.mkdir(parents=True, exist_ok=True)

                for target_name, steer_prompts in target_prompts.items():
                    progress.set_postfix(
                        seed=random_seed,
                        config=description,
                        target=target_name,
                        refresh=True,
                    )
                    (
                        tokenized_text,
                        masking_probs,
                        masked_positions,
                        regenerated_texts,
                    ) = _run_configuration(
                        configuration=configuration,
                        target_name=target_name,
                        steer_prompts=steer_prompts,
                        texts=texts,
                        base_prompts=base_prompts,
                        model=model,
                        tokenizer=tokenizer,
                        identifier_model=identifier_metadata,
                        cached_scores=cached_scores,
                        random_seed=random_seed,
                    )

                    visualize_timpa_probabilistic(
                        tokenized_text,
                        masking_probs,
                        masked_positions,
                        regenerated_texts,
                        output_file=html_directory / f"{target_name}.html",
                    )
                    write_before_after_csv(
                        csv_directory / f"{target_name}.csv",
                        texts,
                        regenerated_texts,
                    )
                    progress.update()

    print(f"Wrote CSV test results to {CSV_ROOT}")
    print(f"Wrote HTML test results to {HTML_ROOT}")


if __name__ == "__main__":
    main()
