import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpa_datasets import timpa_load_data_and_steer_artefacts, timpa_load_rows
from timpateks import timpa_ar, timpa_hybrid, timpa_probabilistic, timpa_steer
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM


METHODS = (
    "timpa_ar",
    "timpa_probabilistic",
    "timpa_steer",
    "timpa_hybrid",
)
DATASETS = (
    "catdog",
    "imdb",
    "elifive",
    "dolly_sample",
)


def _effective_method_parameters(args):
    defaults = {
        "timpa_ar": {
            "temperature": 0.0,
        },
        "timpa_probabilistic": {
            "temperature": 1.0,
            "margin": 0.001,
            "sampling_temperature": 0.0,
        },
        "timpa_steer": {
            "temperature": 1.0,
            "margin": 0.05,
            "sampling_temperature": 1.0,
        },
        "timpa_hybrid": {
            "temperature": 1.0,
            "margin": 0.001,
            "sampling_temperature": 1.0,
        },
    }[args.method]
    effective = dict(defaults)
    for name in ("temperature", "margin", "sampling_temperature"):
        value = getattr(args, name, None)
        if value is not None and name in effective:
            effective[name] = value
    return effective


def _experiment_id(run_name):
    if run_name:
        identifier = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._-")
        if not identifier:
            raise ValueError("--run-name must contain at least one letter or number.")
        return identifier
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _technique_name(args):
    if args.method == "timpa_ar":
        return "ar"
    if args.method == "timpa_steer":
        if args.steer not in {"add", "projection"}:
            raise ValueError("timpa_steer requires steer='add' or steer='projection'.")
        return f"timpa_steer_{args.steer}"
    return args.method


def _save_results(args, results):
    technique = _technique_name(args)
    output_dir = Path(args.output_dir) / args.dataset / technique
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = _experiment_id(args.run_name)
    experiment_name = f"{args.dataset}_{technique}_{experiment_id}"
    csv_path = output_dir / f"{experiment_name}.csv"
    metadata_path = output_dir / f"{experiment_name}.json"
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Experiment output already exists for {experiment_name!r}."
        )

    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("direction", "before", "after1", "picked_tokens"),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "direction": result["target_direction"],
                    "before": result["text"],
                    "after1": result["regenerated_text"],
                    "picked_tokens": json.dumps(
                        result["picked_tokens"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    direction_counts = Counter(result["target_direction"] for result in results)
    metadata = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "technique": technique,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_file": csv_path.name,
        "num_results": len(results),
        "direction_counts": dict(direction_counts),
        "result_columns": ["direction", "before", "after1", "picked_tokens"],
        "effective_method_parameters": _effective_method_parameters(args),
        "arguments": arguments,
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return csv_path, metadata_path


def _make_api_completion_fn(args):
    if not args.api_model_id:
        raise ValueError("--api-model-id is required when --ar-backend api is used.")
    if not args.api_base_url:
        raise ValueError("--api-base-url is required when --ar-backend api is used.")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"The {args.api_key_env!r} environment variable must contain the API key."
        )
    endpoint = f"{args.api_base_url.rstrip('/')}/chat/completions"

    def completion_fn(messages, max_new_tokens, temperature):
        payload = json.dumps(
            {
                "model": args.api_model_id,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AR API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach the AR API: {exc.reason}") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("AR API response did not contain a completion.") from exc
        if not isinstance(content, str):
            raise RuntimeError("AR API completion must be a string.")
        return content

    return completion_fn


def _collate_direction_jobs(dataset_name, dataset, steer_artifact):
    if dataset_name in {"imdb", "catdog"}:
        directions = list(dataset)
        if len(directions) != 2:
            raise ValueError("Paired datasets must contain exactly two directions.")
        return [
            {
                "source_direction": source_direction,
                "target_direction": next(
                    direction
                    for direction in directions
                    if direction != source_direction
                ),
                "texts": texts,
            }
            for source_direction, texts in dataset.items()
        ]

    target_directions = [
        direction for direction in steer_artifact if direction != "base"
    ]
    return [
        {
            "source_direction": "base",
            "target_direction": target_direction,
            "texts": dataset["text"],
        }
        for target_direction in target_directions
    ]


def _optional_generation_args(args, include_margin=True):
    values = {}
    if args.temperature is not None:
        values["temperature"] = args.temperature
    if include_margin and args.margin is not None:
        values["margin"] = args.margin
    if args.sampling_temperature is not None:
        values["sampling_temperature"] = args.sampling_temperature
    return values


def _picked_tokens_from_mask(tokenizer, tokenized_text, masked_positions):
    input_ids = tokenized_text.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("tokenized_text input_ids must have shape [batch, tokens].")
    if (
        not isinstance(masked_positions, torch.Tensor)
        or masked_positions.shape != input_ids.shape
    ):
        raise ValueError("masked_positions must match tokenized_text input_ids.")

    attention_mask = tokenized_text.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match tokenized_text input_ids.")

    picked_tokens = []
    for row in range(input_ids.shape[0]):
        valid_positions = attention_mask[row].bool()
        row_ids = input_ids[row][valid_positions].detach().cpu().tolist()
        row_picks = (
            masked_positions[row][valid_positions].bool().detach().cpu().tolist()
        )
        selected = []
        for position, (token_id, is_picked) in enumerate(zip(row_ids, row_picks)):
            if not is_picked:
                continue
            selected.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id),
                    "text": tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
        picked_tokens.append(selected)
    return picked_tokens


def _run_batch(
    args,
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer_artifact,
    source_direction,
    target_direction,
    texts,
    generator,
    completion_fn,
):
    if args.method == "timpa_ar":
        instruction = steer_artifact[target_direction]
        kwargs = {}
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        regenerated_texts = timpa_ar(
            instruction=[instruction] * len(texts),
            text=texts,
            model=identifier_model,
            tokenizer=identifier_tokenizer,
            completion_fn=completion_fn,
            max_new_tokens=args.max_new_tokens,
            **kwargs,
        )
        return regenerated_texts, [[] for _ in regenerated_texts]

    if args.method == "timpa_probabilistic":
        base_prompt = steer_artifact.get(source_direction, steer_artifact.get("base"))
        target_prompt = steer_artifact[target_direction]
        tokenized_text, _, masked_positions, regenerated_texts = timpa_probabilistic(
            model=model,
            tokenizer=tokenizer,
            identifier_model=identifier_model,
            identifier_tokenizer=identifier_tokenizer,
            steer=[target_prompt] * len(texts),
            text=texts,
            base_assistant_prompt=[base_prompt] * len(texts),
            generator=generator,
            refill_steps=args.refill_steps,
            refill_strategy=args.refill_strategy,
            detection_strategy=args.detection,
            random_mask_probability=args.random_mask_probability,
            **_optional_generation_args(args),
        )
        return regenerated_texts, _picked_tokens_from_mask(
            tokenizer,
            tokenized_text,
            masked_positions,
        )

    if args.method == "timpa_steer":
        tokenized_text, _, masked_positions, regenerated_texts = timpa_steer(
            model=model,
            tokenizer=tokenizer,
            steer_vectors=steer_artifact[target_direction],
            text=texts,
            generator=generator,
            refill_steps=args.refill_steps,
            refill_strategy=args.refill_strategy,
            steer_mode="add" if args.steer == "add" else "project_out",
            alpha=args.alpha,
            detection_strategy=args.detection,
            random_mask_probability=args.random_mask_probability,
            **_optional_generation_args(args),
        )
        return regenerated_texts, _picked_tokens_from_mask(
            tokenizer,
            tokenized_text,
            masked_positions,
        )

    direction_artifact = steer_artifact[target_direction]
    steerprompts = direction_artifact["steerprompts"]
    base_prompt = steerprompts.get(source_direction, steerprompts.get("base"))
    target_prompt = steerprompts[target_direction]
    tokenized_text, _, masked_positions, regenerated_texts = timpa_hybrid(
        model=model,
        tokenizer=tokenizer,
        identifier_model=identifier_model,
        identifier_tokenizer=identifier_tokenizer,
        steer_vectors=direction_artifact["vector"],
        steer=[target_prompt] * len(texts),
        text=texts,
        base_assistant_prompt=[base_prompt] * len(texts),
        generator=generator,
        refill_steps=args.refill_steps,
        refill_strategy=args.refill_strategy,
        alpha=args.alpha,
        **_optional_generation_args(args),
    )
    return regenerated_texts, _picked_tokens_from_mask(
        tokenizer,
        tokenized_text,
        masked_positions,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a TIMPA text-replacement benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--method", choices=METHODS, required=True)
    experiment.add_argument("--dataset", choices=DATASETS, required=True)
    experiment.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="The train split is currently loaded from each dataset's val.csv.",
    )
    experiment.add_argument("--max-samples", type=int)
    experiment.add_argument("--batch-size", type=int, default=1)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--run-name")
    experiment.add_argument(
        "--output-dir",
        type=Path,
        default=Path("timpateks_results"),
        help="Root directory; results are grouped below dataset/technique.",
    )

    models = parser.add_argument_group("models")
    models.add_argument(
        "--dlm-model-id",
        default="GSAI-ML/LLaDA-8B-Instruct",
    )
    models.add_argument(
        "--ar-model-id",
        default="Qwen/Qwen2.5-14B-Instruct",
    )

    ar = parser.add_argument_group("AR baseline")
    ar.add_argument(
        "--ar-backend",
        choices=("local", "api"),
        default="local",
    )
    ar.add_argument("--max-new-tokens", type=int, default=2048)
    ar.add_argument("--api-model-id")
    ar.add_argument("--api-base-url")
    ar.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; do not pass the key itself.",
    )

    detection = parser.add_argument_group("token detection")
    detection.add_argument(
        "--detection",
        choices=("model", "random"),
        default="model",
    )
    detection.add_argument(
        "--temperature",
        type=float,
        help="Leave unset to use the selected method's default.",
    )
    detection.add_argument(
        "--margin",
        type=float,
        help="Leave unset to use the selected method's default.",
    )
    detection.add_argument("--random-mask-probability", type=float, default=0.5)

    refill = parser.add_argument_group("diffusion refill")
    refill.add_argument("--refill-steps", type=int, default=32)
    refill.add_argument(
        "--sampling-temperature",
        type=float,
        help="Leave unset to use the selected method's default.",
    )
    refill.add_argument(
        "--refill-strategy",
        choices=("low_confidence", "random"),
        default="low_confidence",
    )

    steering = parser.add_argument_group("activation steering")
    steering.add_argument(
        "--steer",
        choices=("add", "projection"),
        help="Activation intervention used by timpa_steer.",
    )
    steering.add_argument("--alpha", type=float, default=1.0)
    steering.add_argument(
        "--steer-layers",
        type=int,
        nargs="+",
        default=[16, 25, 31],
    )
    steering.add_argument("--source-layer", type=int, default=23)
    steering.add_argument("--token-position", type=int, default=-4)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    uses_steering = args.method in {"timpa_steer", "timpa_hybrid"}

    if uses_steering and args.dataset not in {"imdb", "catdog"}:
        raise ValueError(
            f"{args.method} does not support the {args.dataset!r} dataset."
        )
    if uses_steering and args.steer is None:
        raise ValueError(
            f"--steer is required for {args.method}; choose 'add' or 'projection'."
        )
    if args.method == "timpa_hybrid" and args.steer != "add":
        raise ValueError("timpa_hybrid requires --steer add.")
    if args.method == "timpa_hybrid" and args.detection != "model":
        raise ValueError("timpa_hybrid requires --detection model.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero.")

    model = None
    tokenizer = None
    if args.method in {"timpa_probabilistic", "timpa_steer", "timpa_hybrid"}:
        config = LLaDAConfig.from_pretrained(args.dlm_model_id)
        model = (
            LLaDAModelLM.from_pretrained(
                args.dlm_model_id,
                config=config,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.dlm_model_id,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"

    identifier_model = None
    identifier_tokenizer = None
    needs_local_ar = (
        args.method == "timpa_hybrid"
        or (
            args.method == "timpa_probabilistic"
            and args.detection == "model"
        )
        or (args.method == "timpa_ar" and args.ar_backend == "local")
    )
    if needs_local_ar:
        identifier_model = (
            AutoModelForCausalLM.from_pretrained(
                args.ar_model_id,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        identifier_tokenizer = AutoTokenizer.from_pretrained(args.ar_model_id)
        identifier_tokenizer.padding_side = "left"

    if uses_steering:
        dataset = timpa_load_rows(args.dataset)["dataset"][args.split]
        steer_artifact = {}
        for steer_direction in dataset:
            _, direction_artifact = timpa_load_data_and_steer_artefacts(
                args.dataset,
                args.split,
                args.method,
                model=model,
                tokenizer=tokenizer,
                steer_direction=steer_direction,
                steer_method=args.steer,
                source_layer=args.source_layer,
                token_position=args.token_position,
                steer_layers=args.steer_layers,
            )
            steer_artifact[steer_direction] = direction_artifact
    else:
        dataset, steer_artifact = timpa_load_data_and_steer_artefacts(
            args.dataset,
            args.split,
            args.method,
            steer_layers=args.steer_layers,
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    completion_fn = None
    if args.method == "timpa_ar" and args.ar_backend == "api":
        completion_fn = _make_api_completion_fn(args)

    jobs = _collate_direction_jobs(args.dataset, dataset, steer_artifact)
    results = []
    for job in jobs:
        texts = job["texts"]
        if args.max_samples is not None:
            texts = texts[:args.max_samples]

        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            regenerated_texts, picked_tokens = _run_batch(
                args=args,
                model=model,
                tokenizer=tokenizer,
                identifier_model=identifier_model,
                identifier_tokenizer=identifier_tokenizer,
                steer_artifact=steer_artifact,
                source_direction=job["source_direction"],
                target_direction=job["target_direction"],
                texts=batch_texts,
                generator=generator,
                completion_fn=completion_fn,
            )
            if len(regenerated_texts) != len(batch_texts):
                raise RuntimeError(
                    f"{args.method} returned {len(regenerated_texts)} texts for a "
                    f"batch containing {len(batch_texts)} inputs."
                )
            if len(picked_tokens) != len(batch_texts):
                raise RuntimeError(
                    f"{args.method} returned token selections for "
                    f"{len(picked_tokens)} texts for a batch containing "
                    f"{len(batch_texts)} inputs."
                )
            for text, regenerated_text, text_picked_tokens in zip(
                batch_texts,
                regenerated_texts,
                picked_tokens,
            ):
                results.append(
                    {
                        "source_direction": job["source_direction"],
                        "target_direction": job["target_direction"],
                        "text": text,
                        "regenerated_text": regenerated_text,
                        "picked_tokens": text_picked_tokens,
                    }
                )

    csv_path, metadata_path = _save_results(args, results)
    print(f"Saved results to {csv_path}")
    print(f"Saved metadata to {metadata_path}")
    return results


if __name__ == "__main__":
    main()
