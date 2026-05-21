from __future__ import annotations
import argparse
from pathlib import Path


def assert_single_sweep_dimension(args: argparse.Namespace) -> None:
    sweep_args = {
        "refill_steps": args.refill_steps,
        "sampling_temp": args.sampling_temp,
        "identify_temp": args.identify_temp,
    }
    multi_value_args = [name for name, values in sweep_args.items() if len(values) > 1]
    assert len(multi_value_args) <= 1, (
        "Only one of refill_steps, sampling_temp, or identify_temp can have "
        f"more than one value. Got multiple values for: {', '.join(multi_value_args)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TIMPA experiment.")
    parser.add_argument("--run-name", "--run_name", dest="run_name", type=str, required=True)
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--random-state", "--random_state", dest="random_state", type=int, required=True)
    parser.add_argument(
        "--steer-vector-path",
        "--steer_vector_path",
        dest="steer_vector_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--steer-alpha", "--steer_alpha", dest="steer_alpha", type=float, required=True)
    parser.add_argument("--steer-layers", "--steer_layers", dest="steer_layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, required=True)
    parser.add_argument("--resteer-steps", "--resteer_steps", dest="resteer_steps", type=int, required=True)
    parser.add_argument("--refill-steps", "--refill_steps", dest="refill_steps", type=int, nargs="+", required=True)
    parser.add_argument("--sampling-temp", "--sampling_temp", dest="sampling_temp", type=float, nargs="+", required=True)
    parser.add_argument("--identify-temp", "--identify_temp", dest="identify_temp", type=float, nargs="+", required=True)
    args = parser.parse_args()
    assert_single_sweep_dimension(args)
    return args
