#!/usr/bin/env python3
"""Benchmark instruction prompting and two TIMPA variants on IMDb."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from timpa_datasets import timpa_load_data_and_steer_artefacts
from timpa_eval.perf_benchmark import (
    BenchmarkRun,
    RunPayload,
    benchmark_method,
    instruction_rewrite_batch,
    masked_fractions,
    mean_source_token_length,
    measure_cuda_phase,
    read_raw_csv,
    seed_everything,
    write_html_report,
    write_raw_csv,
)
from timpateks import helpers, timpa_steer
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEVICE = "cuda"
LLADA_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
IDENTIFIER_MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
LOCAL_FILES_ONLY = True

TARGET_DIRECTIONS = ("positive", "negative")
STEER_LAYERS = (20, 31)
STEER_ALPHA = 1.0
STEER_DETECTION_TEMPERATURE = 3.0
STEER_MARGIN = 0.05
PROBABILISTIC_TEMPERATURE = 0.5
PROBABILISTIC_MARGIN = 0.1

BATCH_SIZE = 5
REFILL_STEPS = 32
SAMPLING_TEMPERATURE = 0.5
REFILL_STRATEGY = "low_confidence"
INSTRUCTION_GENERATION_LENGTH = 128
MEASURED_SEEDS = (42, 43, 44)

OUTPUT_HTML = Path("main_results/imdb_perf_benchmark.html")
OUTPUT_CSV = Path("main_results/imdb_perf_benchmark_raw.csv")


@dataclass(frozen=True)
class BenchmarkBatch:
    source_direction: str
    target_direction: str
    texts: tuple[str, ...]


@dataclass(frozen=True)
class ProbDetectedBatch:
    batch: BenchmarkBatch
    attention_mask: torch.Tensor
    masked_positions: torch.Tensor


@dataclass(frozen=True)
class ProbDetectionPayload:
    batches: tuple[ProbDetectedBatch, ...]
    masked_fractions: tuple[float, ...]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-per-direction",
        type=int,
        default=100,
        help="IMDb examples per steering direction (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=len(MEASURED_SEEDS),
        help="Measured repetitions after one warmup batch (default: 3).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate data/configuration without loading either model.",
    )
    parser.add_argument(
        "--probabilistic-only",
        action="store_true",
        help=(
            "Reuse checkpointed instruction/steering CSV rows and run only "
            "the staged probabilistic benchmark."
        ),
    )
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    return parser.parse_args()


def _load_llada():
    config = LLaDAConfig.from_pretrained(
        LLADA_MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = (
        LLaDAModelLM.from_pretrained(
            LLADA_MODEL_ID,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=LOCAL_FILES_ONLY,
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LLADA_MODEL_ID,
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def _load_identifier():
    model = (
        AutoModelForCausalLM.from_pretrained(
            IDENTIFIER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=LOCAL_FILES_ONLY,
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        IDENTIFIER_MODEL_ID,
        local_files_only=LOCAL_FILES_ONLY,
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def _opposite(target_direction):
    return next(
        direction
        for direction in TARGET_DIRECTIONS
        if direction != target_direction
    )


def _prepare_batches(examples_per_direction, batch_size):
    dataset, prompts = timpa_load_data_and_steer_artefacts(
        "imdb",
        "test",
        "timpa_probabilistic",
    )
    batches = []
    all_texts = []
    for target_direction in TARGET_DIRECTIONS:
        source_direction = _opposite(target_direction)
        texts = dataset[source_direction]
        if examples_per_direction > len(texts):
            raise ValueError(
                f"Requested {examples_per_direction} {source_direction} examples, "
                f"but only {len(texts)} are available."
            )
        texts = texts[:examples_per_direction]
        all_texts.extend(texts)
        for start in range(0, len(texts), batch_size):
            batches.append(
                BenchmarkBatch(
                    source_direction=source_direction,
                    target_direction=target_direction,
                    texts=tuple(texts[start:start + batch_size]),
                )
            )
    return batches, all_texts, prompts


def _extract_steering_vectors(model, tokenizer):
    _, positive_vectors = timpa_load_data_and_steer_artefacts(
        dataset_name="imdb",
        split="test",
        timpa_method="timpa_steer",
        model=model,
        tokenizer=tokenizer,
        steer_direction="positive",
        steer_method="add",
        steer_layers=STEER_LAYERS,
    )
    return {
        "positive": positive_vectors,
        "negative": {
            layer: -vector for layer, vector in positive_vectors.items()
        },
    }


def _progress(batches, description, enabled):
    return tqdm(
        batches,
        desc=description,
        unit="batch",
        leave=False,
        disable=not enabled,
    )


def _run_instruction(model, tokenizer, batches, seed, show_progress=True):
    del seed
    for batch in _progress(batches, "Instruction baseline", show_progress):
        instruction = (
            f"Change this movie review to a {batch.target_direction} movie "
            "review while preserving the original content."
        )
        rewritten = instruction_rewrite_batch(
            model=model,
            tokenizer=tokenizer,
            instructions=[instruction] * len(batch.texts),
            texts=batch.texts,
            steps=REFILL_STEPS,
            generation_length=INSTRUCTION_GENERATION_LENGTH,
            sampling_temperature=SAMPLING_TEMPERATURE,
            refill_strategy=REFILL_STRATEGY,
        )
        if len(rewritten) != len(batch.texts):
            raise RuntimeError("Instruction baseline returned the wrong batch size.")
    return RunPayload()


def _run_steering(
    model,
    tokenizer,
    vectors,
    batches,
    seed,
    show_progress=True,
):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    fractions = []
    for batch in _progress(batches, "TIMPA steering", show_progress):
        tokenized, _, positions, rewritten = timpa_steer(
            model=model,
            tokenizer=tokenizer,
            steer_vectors=vectors[batch.target_direction],
            text=list(batch.texts),
            refill_steps=REFILL_STEPS,
            sampling_temperature=SAMPLING_TEMPERATURE,
            temperature=STEER_DETECTION_TEMPERATURE,
            generator=generator,
            refill_strategy=REFILL_STRATEGY,
            system_prompt="You are a helpful assistant.",
            use_chat_template=True,
            steer_mode="add",
            alpha=STEER_ALPHA,
            margin=STEER_MARGIN,
            detection_strategy="model",
        )
        if len(rewritten) != len(batch.texts):
            raise RuntimeError("TIMPA steering returned the wrong batch size.")
        fractions.extend(
            masked_fractions(positions, tokenized["attention_mask"])
        )
    return RunPayload(masked_fractions=tuple(fractions))


def _detect_probabilistic(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    prompts,
    batches,
    seed,
    show_progress=True,
):
    del model
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    fractions = []
    detected_batches = []
    for batch in _progress(batches, "Probabilistic detection", show_progress):
        tokenized, aligned_deltas = helpers._probabilistic_token_scores(
            tokenizer=tokenizer,
            identifier_model=identifier_model,
            identifier_tokenizer=identifier_tokenizer,
            steer=[prompts[batch.target_direction]] * len(batch.texts),
            texts=list(batch.texts),
            base_assistant_prompt=[prompts[batch.source_direction]]
            * len(batch.texts),
            use_chat_template=True,
        )
        tokenized, _, positions = (
            helpers._probabilistic_token_detection_from_scores(
                tokenizer=tokenizer,
                texts=list(batch.texts),
                tokenized_text=tokenized,
                aligned_word_log_deltas=aligned_deltas,
                temperature=PROBABILISTIC_TEMPERATURE,
                margin=PROBABILISTIC_MARGIN,
                mask_selection="sample",
                generator=generator,
            )
        )
        attention_mask = tokenized["attention_mask"]
        fractions.extend(masked_fractions(positions, attention_mask))
        detected_batches.append(
            ProbDetectedBatch(
                batch=batch,
                attention_mask=attention_mask.detach().cpu(),
                masked_positions=positions.detach().cpu(),
            )
        )
    return ProbDetectionPayload(
        batches=tuple(detected_batches),
        masked_fractions=tuple(fractions),
    )


def _refill_probabilistic(
    model,
    tokenizer,
    prompts,
    detection,
    show_progress=True,
):
    for detected in _progress(
        detection.batches,
        "Probabilistic refill",
        show_progress,
    ):
        batch = detected.batch
        rewritten = helpers.regenerate_masked_text(
            model=model,
            tokenizer=tokenizer,
            steer=[prompts[batch.target_direction]] * len(batch.texts),
            text=list(batch.texts),
            masked_positions=detected.masked_positions.to(DEVICE),
            response_attention_mask=detected.attention_mask.to(DEVICE),
            use_chat_template=True,
            refill_steps=REFILL_STEPS,
            sampling_temperature=SAMPLING_TEMPERATURE,
            refill_strategy=REFILL_STRATEGY,
        )
        if len(rewritten) != len(batch.texts):
            raise RuntimeError("TIMPA probabilistic returned the wrong batch size.")
    return RunPayload(masked_fractions=detection.masked_fractions)


def _benchmark_probabilistic_staged(
    model,
    tokenizer,
    prompts,
    batches,
    example_count,
    input_tokens,
    measured_seeds,
):
    print("Loading Qwen identifier outside the timed region...")
    identifier_model, identifier_tokenizer = _load_identifier()
    torch.cuda.empty_cache()

    seed_everything(0)
    warm_detection = _detect_probabilistic(
        model,
        tokenizer,
        identifier_model,
        identifier_tokenizer,
        prompts,
        batches[:1],
        0,
        False,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    detection_measurements = []
    detected_runs = []
    print("Benchmarking uncached probabilistic detection...")
    for repetition, seed in enumerate(measured_seeds, start=1):
        seed_everything(seed)
        detected, seconds, peak, incremental = measure_cuda_phase(
            lambda seed=seed: _detect_probabilistic(
                model,
                tokenizer,
                identifier_model,
                identifier_tokenizer,
                prompts,
                batches,
                seed,
            ),
            device=DEVICE,
        )
        detected_runs.append(detected)
        detection_measurements.append(
            (repetition, seed, seconds, peak, incremental)
        )

    del identifier_model, identifier_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Released Qwen; benchmarking probabilistic LLaDA refill...")

    seed_everything(0)
    _refill_probabilistic(
        model,
        tokenizer,
        prompts,
        warm_detection,
        False,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    observations = []
    for detected, detection_metrics in zip(
        detected_runs,
        detection_measurements,
    ):
        (
            repetition,
            seed,
            detection_seconds,
            detection_peak,
            detection_incremental,
        ) = detection_metrics
        seed_everything(seed)
        _, refill_seconds, refill_peak, refill_incremental = measure_cuda_phase(
            lambda detected=detected: _refill_probabilistic(
                model,
                tokenizer,
                prompts,
                detected,
            ),
            device=DEVICE,
        )
        elapsed = detection_seconds + refill_seconds
        observations.append(
            BenchmarkRun(
                method="TIMPA probabilistic",
                repetition=repetition,
                seed=seed,
                examples=example_count,
                elapsed_seconds=elapsed,
                latency_ms_per_example=1000.0 * elapsed / example_count,
                examples_per_second=example_count / elapsed,
                peak_allocated_gib=max(detection_peak, refill_peak),
                incremental_peak_gib=max(
                    detection_incremental,
                    refill_incremental,
                ),
                mean_input_tokens=input_tokens,
                mean_masked_fraction=(
                    sum(detected.masked_fractions)
                    / len(detected.masked_fractions)
                ),
            )
        )
    return observations


def main():
    args = _parse_args()
    if args.examples_per_direction <= 0:
        raise ValueError("--examples-per-direction must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.repetitions <= len(MEASURED_SEEDS):
        raise ValueError(
            f"--repetitions must be between 1 and {len(MEASURED_SEEDS)}."
        )

    batches, all_texts, prompts = _prepare_batches(
        args.examples_per_direction,
        args.batch_size,
    )
    example_count = len(all_texts)
    print(
        f"Prepared {example_count} IMDb examples in {len(batches)} batches "
        f"({args.examples_per_direction} per direction)."
    )
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The performance benchmark requires CUDA.")

    measured_seeds = MEASURED_SEEDS[:args.repetitions]
    model, tokenizer = _load_llada()
    input_tokens = mean_source_token_length(tokenizer, all_texts)
    steering_vectors = (
        None
        if args.probabilistic_only
        else _extract_steering_vectors(model, tokenizer)
    )
    torch.cuda.empty_cache()

    results = read_raw_csv(args.output_csv) if args.probabilistic_only else []
    if args.probabilistic_only:
        expected_methods = {"Instruction prompting", "TIMPA steering"}
        present_methods = {run.method for run in results}
        if present_methods != expected_methods or len(results) != 6:
            raise RuntimeError(
                "--probabilistic-only requires exactly six checkpointed "
                "instruction/steering rows."
            )
        if any(run.examples != example_count for run in results):
            raise RuntimeError("Checkpointed runs use a different example count.")

    if not args.probabilistic_only:
        print("Benchmarking instruction prompting...")
        results.extend(
            benchmark_method(
                method="Instruction prompting",
                example_count=example_count,
                mean_input_tokens=input_tokens,
                warmup=lambda seed: _run_instruction(
                    model, tokenizer, batches[:1], seed, False
                ),
                measured_run=lambda seed: _run_instruction(
                    model, tokenizer, batches, seed
                ),
                seeds=measured_seeds,
                device=DEVICE,
            )
        )
        write_raw_csv(args.output_csv, results)

        print("Benchmarking TIMPA steering...")
        results.extend(
            benchmark_method(
                method="TIMPA steering",
                example_count=example_count,
                mean_input_tokens=input_tokens,
                warmup=lambda seed: _run_steering(
                    model,
                    tokenizer,
                    steering_vectors,
                    batches[:1],
                    seed,
                    False,
                ),
                measured_run=lambda seed: _run_steering(
                    model,
                    tokenizer,
                    steering_vectors,
                    batches,
                    seed,
                ),
                seeds=measured_seeds,
                device=DEVICE,
            )
        )
        write_raw_csv(args.output_csv, results)

    results = [
        run for run in results if run.method != "TIMPA probabilistic"
    ]
    results.extend(
        _benchmark_probabilistic_staged(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batches=batches,
            example_count=example_count,
            input_tokens=input_tokens,
            measured_seeds=measured_seeds,
        )
    )
    write_raw_csv(args.output_csv, results)

    hardware = torch.cuda.get_device_name(torch.cuda.current_device())
    write_html_report(
        args.output_html,
        results,
        title="IMDb TIMPA Performance Benchmark",
        hardware=hardware,
        method_details={
            "Instruction prompting": (
                "Direct LLaDA rewrite instruction; 128 generated tokens, "
                "32 diffusion steps, sampling temperature 0.5."
            ),
            "TIMPA steering": (
                "Additive vectors at layers [20, 31], alpha 1, detection "
                "temperature 3, margin 0.05, sampling temperature 0.5."
            ),
            "TIMPA probabilistic": (
                "Uncached Qwen-32B base/target scoring, detection temperature "
                "0.5, margin 0.1, sampling temperature 0.5. Detection and refill "
                "are phase-staged so Qwen is released before LLaDA sampling."
            ),
        },
        methodology=[
            f"{example_count} IMDb test examples: equal positive-to-negative "
            "and negative-to-positive directions.",
            f"Batch size {args.batch_size}; one excluded warmup batch per method; "
            f"measured seeds {list(measured_seeds)}.",
            "End-to-end timing includes input tokenization, detection, mask "
            "selection, LLaDA generation/refill, and decoding.",
            "Model/tokenizer loading, steering-vector extraction, dataset loading, "
            "file writing, visualization, and evaluation are excluded.",
            "CUDA is synchronized immediately before and after each measured run.",
            "Probabilistic detection is recomputed for every repetition; sweep "
            "score caches are not reused.",
            "On the 80 GiB A100, probabilistic detection and refill are measured "
            "as consecutive stages: all uncached detection runs are measured with "
            "Qwen and LLaDA resident, Qwen is released, then matching refill runs "
            "are measured. Per-repetition phase times are summed and phase peaks "
            "are combined by maximum.",
            "Peak memory is torch.cuda.max_memory_allocated with the models required "
            "by each measured phase already resident.",
        ],
    )
    print(f"Wrote raw benchmark data to {args.output_csv}")
    print(f"Wrote HTML benchmark report to {args.output_html}")


if __name__ == "__main__":
    main()
