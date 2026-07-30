import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from timpa_datasets import (
    timpa_load_data_and_steer_artefacts,
    timpa_load_rows,
)
from timpateks import timpa_steer
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM

from .core import visualize_timpa_steer_results


@dataclass(frozen=True)
class SteerVectorSweepConfig:
    dataset_name: str
    target_directions: tuple[str, str]
    output_csv_root: Path
    output_html_root: Path
    model_id: str
    device: str = "cuda"
    local_files_only: bool = True
    split: str = "train"
    layer_candidates: tuple[int, ...] | None = None
    add_alphas: tuple[float, ...] = (100.0, 300.0, 600.0, 900.0, 1200.0)
    random_seeds: tuple[int, ...] = (42,)
    generation_batch_size: int = 5
    refill_steps: int = 32
    sampling_temperature: float = 0.1
    random_mask_probability: float = 0.5
    refill_strategy: str = "low_confidence"
    system_prompt: str = "You are a helpful assistant."


def _validate_config(config):
    if len(set(config.target_directions)) != 2:
        raise ValueError("target_directions must contain two distinct concepts.")
    for name, value in (
        ("generation_batch_size", config.generation_batch_size),
        ("refill_steps", config.refill_steps),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not config.random_seeds or not all(
        isinstance(seed, int) for seed in config.random_seeds
    ):
        raise ValueError("random_seeds must contain at least one integer.")
    if not config.add_alphas or any(value <= 0 for value in config.add_alphas):
        raise ValueError("add_alphas must contain positive values.")
    if not 0 <= config.random_mask_probability <= 1:
        raise ValueError("random_mask_probability must be between zero and one.")


def _seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(config):
    dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
    model_config = LLaDAConfig.from_pretrained(
        config.model_id,
        local_files_only=config.local_files_only,
    )
    model = (
        LLaDAModelLM.from_pretrained(
            config.model_id,
            config=model_config,
            torch_dtype=dtype,
            local_files_only=config.local_files_only,
        )
        .to(config.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    tokenizer.padding_side = "left"

    return model, tokenizer


def _resolve_layers(model, configured_layers):
    num_layers = getattr(model.config, "n_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("The LLaDA configuration must define n_layers.")
    if configured_layers is None:
        return tuple(range(num_layers))

    layers = tuple(dict.fromkeys(configured_layers))
    if not layers or any(
        not isinstance(layer, int) or not 0 <= layer < num_layers
        for layer in layers
    ):
        raise ValueError(
            f"layer_candidates must be transformer layers from 0 to "
            f"{num_layers - 1}."
        )
    return layers


def _opposite_vectors(vectors):
    return {
        layer: -vector
        for layer, vector in vectors.items()
    }


def _extract_add_vectors(config, model, tokenizer, layers):
    first_target, second_target = config.target_directions
    _, first_vectors = timpa_load_data_and_steer_artefacts(
        dataset_name=config.dataset_name,
        split=config.split,
        timpa_method="timpa_steer",
        model=model,
        tokenizer=tokenizer,
        steer_direction=first_target,
        steer_method="add",
        steer_layers=layers,
    )
    return {
        first_target: first_vectors,
        second_target: _opposite_vectors(first_vectors),
    }


def _run_direction(
    config,
    model,
    tokenizer,
    steer_vectors,
    alpha,
    source_texts,
    target_direction,
    seed,
):
    direction_offset = config.target_directions.index(target_direction)
    run_seed = seed + direction_offset
    _seed_everything(run_seed)
    generator = torch.Generator(device=config.device).manual_seed(run_seed)

    regenerated_texts = []
    masked_fractions = []
    probability_rows = []
    position_rows = []
    for start in range(0, len(source_texts), config.generation_batch_size):
        batch = source_texts[start:start + config.generation_batch_size]
        tokenized_text, masking_probs, masked_positions, regenerated = timpa_steer(
            model=model,
            tokenizer=tokenizer,
            steer_vectors=steer_vectors,
            text=batch,
            refill_steps=config.refill_steps,
            sampling_temperature=config.sampling_temperature,
            generator=generator,
            refill_strategy=config.refill_strategy,
            system_prompt=config.system_prompt,
            use_chat_template=True,
            steer_mode="add",
            alpha=alpha,
            detection_strategy="random",
            random_mask_probability=config.random_mask_probability,
        )
        attention_mask = tokenized_text["attention_mask"].bool()
        active_masks = masked_positions.bool() & attention_mask
        masked_fractions.extend(
            (
                active_masks.sum(dim=1)
                / attention_mask.sum(dim=1).clamp_min(1)
            )
            .float()
            .cpu()
            .tolist()
        )
        for row in range(len(batch)):
            active_tokens = attention_mask[row]
            probability_rows.append(
                masking_probs[row][active_tokens].detach().float().cpu()
            )
            position_rows.append(
                masked_positions[row][active_tokens].detach().bool().cpu()
            )
        regenerated_texts.extend(regenerated)

    rows = []
    for example_index, (before, after, masked_fraction) in enumerate(
        zip(
            source_texts,
            regenerated_texts,
            masked_fractions,
        ),
        start=1,
    ):
        rows.append(
            {
                "example_id": example_index,
                "seed": seed,
                "target_direction": target_direction,
                "source_direction": next(
                    direction
                    for direction in config.target_directions
                    if direction != target_direction
                ),
                "text_before": before,
                "text_after": after,
                "masked_token_fraction": masked_fraction,
            }
        )
    visualization_tokens = tokenizer(
        source_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    visualization_attention = visualization_tokens["attention_mask"].bool()
    visualization_probs = torch.zeros(
        visualization_attention.shape,
        dtype=torch.float32,
    )
    visualization_positions = torch.zeros(
        visualization_attention.shape,
        dtype=torch.bool,
    )
    for row, (probabilities_row, positions_row) in enumerate(
        zip(probability_rows, position_rows)
    ):
        active_tokens = visualization_attention[row]
        if active_tokens.sum().item() != probabilities_row.numel():
            raise RuntimeError(
                "Random-mask visualization no longer matches the source tokens."
            )
        visualization_probs[row][active_tokens] = probabilities_row
        visualization_positions[row][active_tokens] = positions_row

    visualization = {
        "seed": seed,
        "target_direction": target_direction,
        "texts": source_texts,
        "tokenized_text": visualization_tokens,
        "masking_probs": visualization_probs,
        "masked_positions": visualization_positions,
        "regenerated_texts": regenerated_texts,
        "steer_vectors": steer_vectors,
    }
    return rows, visualization


def _candidate_rows(
    config,
    model,
    tokenizer,
    dataset,
    vectors_by_target,
    layer,
    alpha,
):
    rows = []
    visualizations = []
    for seed in config.random_seeds:
        for target_direction in config.target_directions:
            source_direction = next(
                direction
                for direction in config.target_directions
                if direction != target_direction
            )
            direction_rows, visualization = _run_direction(
                config=config,
                model=model,
                tokenizer=tokenizer,
                steer_vectors=vectors_by_target[target_direction],
                alpha=alpha,
                source_texts=dataset[source_direction],
                target_direction=target_direction,
                seed=seed,
            )
            for row in direction_rows:
                row.update(
                    {
                        "layer": layer,
                        "alpha": alpha,
                    }
                )
            rows.extend(direction_rows)
            visualizations.append(visualization)
    return rows, visualizations


def _atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _direction_fieldnames():
    return [
        "text_before",
        "text_after",
        "example_id",
        "seed",
        "target_direction",
        "source_direction",
        "masked_token_fraction",
        "layer",
        "alpha",
    ]


def _float_description(value):
    return f"{value:g}"


def _configuration_description(layer, alpha):
    return f"add_layer{layer}_alpha{_float_description(alpha)}"


def _candidate_output_paths(config, description):
    for seed in config.random_seeds:
        for direction in config.target_directions:
            yield (
                config.output_csv_root
                / f"seed{seed}"
                / description
                / f"{direction}.csv"
            )
            yield (
                config.output_html_root
                / f"seed{seed}"
                / description
                / f"{direction}.html"
            )


def _write_candidate_outputs(
    config,
    model,
    tokenizer,
    layer,
    alpha,
    rows,
    visualizations,
):
    description = _configuration_description(
        layer,
        alpha,
    )
    for visualization in visualizations:
        seed = visualization["seed"]
        direction = visualization["target_direction"]
        selected_rows = [
            row for row in rows
            if row["seed"] == seed and row["target_direction"] == direction
        ]
        csv_path = (
            config.output_csv_root
            / f"seed{seed}"
            / description
            / f"{direction}.csv"
        )
        html_path = (
            config.output_html_root
            / f"seed{seed}"
            / description
            / f"{direction}.html"
        )
        _atomic_write_csv(
            csv_path,
            _direction_fieldnames(),
            selected_rows,
        )
        visualize_timpa_steer_results(
            tokenizer=tokenizer,
            text=visualization["texts"],
            tokenized_text=visualization["tokenized_text"],
            masking_probs=visualization["masking_probs"],
            masked_positions=visualization["masked_positions"],
            regenerated_texts=visualization["regenerated_texts"],
            steer_vectors=visualization["steer_vectors"],
            model=model,
            refill_steps=config.refill_steps,
            output_file=html_path,
            system_prompt=config.system_prompt,
            use_chat_template=True,
            steer_mode="add",
            alpha=alpha,
            detection_strategy="random",
            random_mask_probability=config.random_mask_probability,
        )


def _run_add_sweep(
    config,
    model,
    tokenizer,
    dataset,
    candidates,
    vector_factory,
):
    for candidate in tqdm(
        candidates,
        desc=f"{config.dataset_name} add",
        unit="configuration",
    ):
        layer, alpha = candidate
        description = _configuration_description(layer, alpha)
        if all(
            path.is_file()
            for path in _candidate_output_paths(config, description)
        ):
            continue

        vectors_by_target = vector_factory(layer)
        candidate_details, visualizations = _candidate_rows(
            config=config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            vectors_by_target=vectors_by_target,
            layer=layer,
            alpha=alpha,
        )
        _write_candidate_outputs(
            config=config,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            alpha=alpha,
            rows=candidate_details,
            visualizations=visualizations,
        )


def run_steer_vector_sweep(config):
    """Generate raw additive layer-alpha sweep outputs on a validation split."""
    _validate_config(config)
    model, tokenizer = _load_model(config)
    dataset = timpa_load_rows(config.dataset_name)["dataset"][config.split]
    if set(dataset) != set(config.target_directions):
        raise ValueError(
            "The configured target directions do not match the dataset concepts."
        )

    layers = _resolve_layers(model, config.layer_candidates)
    add_vectors = _extract_add_vectors(
        config=config,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
    )

    add_candidates = [
        (layer, alpha)
        for layer in layers
        for alpha in config.add_alphas
    ]
    print(
        f"{config.dataset_name}: {len(add_candidates)} additive configurations."
    )

    _run_add_sweep(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        candidates=add_candidates,
        vector_factory=lambda _layer: {
            target: {_layer: vectors[_layer]}
            for target, vectors in add_vectors.items()
        },
    )
    print(f"Wrote steering-vector CSV results to {config.output_csv_root}")
    print(f"Wrote steering-vector HTML results to {config.output_html_root}")
