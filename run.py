import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpa_datasets import timpa_load_data_and_steer_artefacts, timpa_load_rows
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
    experiment.add_argument("--output-dir", type=Path, default=Path("timpa_results"))

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
    steering.add_argument("--steer-vector-path", type=Path)
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
        raise ValueError(f"--steer is required for {args.method}.") # this is add/projection right? add what options the users can use
    if args.method == "timpa_hybrid" and args.steer != "add":
        raise ValueError("timpa_hybrid requires --steer add.")

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


if __name__ == "__main__":
    main()
