"""Reusable CUDA performance benchmarking helpers for TIMPA experiments."""

from __future__ import annotations

import csv
import html
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence, TypeVar

import torch

from timpateks import helpers
from timpateks.llada.generate import generate as llada_generate


GIB = 1024**3
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class BenchmarkRun:
    method: str
    repetition: int
    seed: int
    examples: int
    elapsed_seconds: float
    latency_ms_per_example: float
    examples_per_second: float
    peak_allocated_gib: float
    incremental_peak_gib: float
    mean_input_tokens: float
    mean_masked_fraction: float | None


@dataclass(frozen=True)
class RunPayload:
    """Small result returned by a benchmark workload after outputs are discarded."""

    masked_fractions: tuple[float, ...] = ()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_fractions(
    masked_positions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[float]:
    active = attention_mask.bool()
    selected = masked_positions.bool() & active
    return (
        selected.sum(dim=1).float()
        .div(active.sum(dim=1).clamp_min(1))
        .detach()
        .cpu()
        .tolist()
    )


def mean_source_token_length(tokenizer, texts: Sequence[str]) -> float:
    lengths = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in texts
    ]
    if not lengths:
        raise ValueError("At least one source text is required.")
    return statistics.fmean(lengths)


def measure_cuda_phase(
    function: Callable[[], PayloadT],
    *,
    device: str = "cuda",
) -> tuple[PayloadT, float, float, float]:
    """Measure one synchronized phase and return payload, time, and memory GiB."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested, but CUDA is unavailable.")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_allocated = 0

    started = perf_counter()
    payload = function()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("Measured elapsed time must be positive.")
    peak_allocated = (
        torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    )
    return (
        payload,
        elapsed,
        peak_allocated / GIB,
        max(0.0, (peak_allocated - baseline_allocated) / GIB),
    )


def benchmark_method(
    *,
    method: str,
    example_count: int,
    mean_input_tokens: float,
    warmup: Callable[[int], RunPayload],
    measured_run: Callable[[int], RunPayload],
    seeds: Sequence[int] = (42, 43, 44),
    device: str = "cuda",
) -> list[BenchmarkRun]:
    """Warm up once, then measure synchronized wall time and CUDA allocation."""
    if not method:
        raise ValueError("method must be non-empty.")
    if example_count <= 0:
        raise ValueError("example_count must be positive.")
    if not seeds:
        raise ValueError("At least one measured seed is required.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested, but CUDA is unavailable.")

    seed_everything(0)
    warmup(0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    observations = []
    for repetition, seed in enumerate(seeds, start=1):
        seed_everything(seed)
        payload, elapsed, peak_gib, incremental_gib = measure_cuda_phase(
            lambda: measured_run(seed),
            device=device,
        )
        fractions = list(payload.masked_fractions)
        if fractions and len(fractions) != example_count:
            raise RuntimeError(
                f"{method} returned {len(fractions)} mask fractions for "
                f"{example_count} examples."
            )
        observations.append(
            BenchmarkRun(
                method=method,
                repetition=repetition,
                seed=seed,
                examples=example_count,
                elapsed_seconds=elapsed,
                latency_ms_per_example=1000.0 * elapsed / example_count,
                examples_per_second=example_count / elapsed,
                peak_allocated_gib=peak_gib,
                incremental_peak_gib=incremental_gib,
                mean_input_tokens=mean_input_tokens,
                mean_masked_fraction=(
                    statistics.fmean(fractions) if fractions else None
                ),
            )
        )
    return observations


def _instruction_prompt_ids(tokenizer, instruction: str, text: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    return prompt_ids[0]


@torch.no_grad()
def instruction_rewrite_batch(
    *,
    model,
    tokenizer,
    instructions: Sequence[str],
    texts: Sequence[str],
    steps: int = 32,
    generation_length: int = 128,
    sampling_temperature: float = 0.5,
    refill_strategy: str = "low_confidence",
) -> list[str]:
    """Direct LLaDA instruction rewriting with masked-diffusion continuation."""
    if len(instructions) != len(texts) or not texts:
        raise ValueError("instructions and texts must be equally sized and non-empty.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The LLaDA tokenizer must define a chat template.")

    sequences = [
        _instruction_prompt_ids(tokenizer, instruction, text)
        for instruction, text in zip(instructions, texts)
    ]
    max_length = max(sequence.numel() for sequence in sequences)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id for batching.")
    device = helpers._model_device(model)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        start = max_length - sequence.numel()
        input_ids[row, start:] = sequence.to(device)
        attention_mask[row, start:] = 1

    generated = llada_generate(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=generation_length,
        block_length=generation_length,
        temperature=sampling_temperature,
        remasking=refill_strategy,
        mask_id=helpers._get_mask_token_id(tokenizer),
    )
    continuations = generated[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(
        continuations,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def write_raw_csv(path: Path, runs: Iterable[BenchmarkRun]) -> None:
    rows = [asdict(run) for run in runs]
    if not rows:
        raise ValueError("At least one benchmark run is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_raw_csv(path: Path) -> list[BenchmarkRun]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    runs = []
    for row in rows:
        masked = row["mean_masked_fraction"]
        runs.append(
            BenchmarkRun(
                method=row["method"],
                repetition=int(row["repetition"]),
                seed=int(row["seed"]),
                examples=int(row["examples"]),
                elapsed_seconds=float(row["elapsed_seconds"]),
                latency_ms_per_example=float(row["latency_ms_per_example"]),
                examples_per_second=float(row["examples_per_second"]),
                peak_allocated_gib=float(row["peak_allocated_gib"]),
                incremental_peak_gib=float(row["incremental_peak_gib"]),
                mean_input_tokens=float(row["mean_input_tokens"]),
                mean_masked_fraction=(float(masked) if masked else None),
            )
        )
    return runs


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def summarize_runs(runs: Sequence[BenchmarkRun]) -> list[dict[str, object]]:
    methods = list(dict.fromkeys(run.method for run in runs))
    summary = []
    for method in methods:
        subset = [run for run in runs if run.method == method]
        latency = _mean_std([run.latency_ms_per_example for run in subset])
        throughput = _mean_std([run.examples_per_second for run in subset])
        peak = _mean_std([run.peak_allocated_gib for run in subset])
        incremental = _mean_std([run.incremental_peak_gib for run in subset])
        mask_values = [
            run.mean_masked_fraction
            for run in subset
            if run.mean_masked_fraction is not None
        ]
        summary.append(
            {
                "method": method,
                "repetitions": len(subset),
                "latency_mean": latency[0],
                "latency_std": latency[1],
                "throughput_mean": throughput[0],
                "throughput_std": throughput[1],
                "peak_mean": peak[0],
                "peak_std": peak[1],
                "incremental_mean": incremental[0],
                "incremental_std": incremental[1],
                "mean_input_tokens": subset[0].mean_input_tokens,
                "masked_fraction": (
                    statistics.fmean(mask_values) if mask_values else None
                ),
            }
        )
    return summary


def write_html_report(
    path: Path,
    runs: Sequence[BenchmarkRun],
    *,
    title: str,
    hardware: str,
    methodology: Sequence[str],
    method_details: dict[str, str],
) -> None:
    if not runs:
        raise ValueError("At least one benchmark run is required.")
    summary = summarize_runs(runs)
    fastest = max(item["throughput_mean"] for item in summary)
    lowest_memory = min(item["peak_mean"] for item in summary)

    summary_rows = []
    for item in summary:
        mask = item["masked_fraction"]
        badges = []
        if item["throughput_mean"] == fastest:
            badges.append('<span class="badge">fastest</span>')
        if item["peak_mean"] == lowest_memory:
            badges.append('<span class="badge">lowest memory</span>')
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(item['method']))}</strong> {' '.join(badges)}</td>"
            f"<td>{item['latency_mean']:.2f} &plusmn; {item['latency_std']:.2f}</td>"
            f"<td>{item['throughput_mean']:.3f} &plusmn; {item['throughput_std']:.3f}</td>"
            f"<td>{item['peak_mean']:.2f} &plusmn; {item['peak_std']:.2f}</td>"
            f"<td>{item['incremental_mean']:.2f} &plusmn; {item['incremental_std']:.2f}</td>"
            f"<td>{item['mean_input_tokens']:.1f}</td>"
            f"<td>{'N/A' if mask is None else f'{100.0 * mask:.2f}%'}</td>"
            "</tr>"
        )

    raw_rows = []
    for run in runs:
        raw_rows.append(
            "<tr>"
            f"<td>{html.escape(run.method)}</td>"
            f"<td>{run.repetition}</td><td>{run.seed}</td>"
            f"<td>{run.elapsed_seconds:.3f}</td>"
            f"<td>{run.latency_ms_per_example:.2f}</td>"
            f"<td>{run.examples_per_second:.3f}</td>"
            f"<td>{run.peak_allocated_gib:.2f}</td>"
            f"<td>{run.incremental_peak_gib:.2f}</td>"
            "</tr>"
        )

    detail_items = "".join(
        f"<li><strong>{html.escape(name)}:</strong> {html.escape(detail)}</li>"
        for name, detail in method_details.items()
    )
    methodology_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in methodology
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#617181; --line:#d9e0e7;
  --panel:#fff; --accent:#146c94; --soft:#eef7fb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#f5f7f9; color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:40px 24px 64px; }}
h1 {{ margin:0 0 6px; font-size:32px; }} h2 {{ margin:34px 0 12px; }}
.subtitle {{ color:var(--muted); margin:0 0 24px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:20px; box-shadow:0 3px 14px rgba(30,55,75,.06); }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:11px 10px; border-bottom:1px solid var(--line); }}
th:first-child,td:first-child {{ text-align:left; }} th {{ color:#425363; font-size:13px; }}
.scroll {{ overflow-x:auto; }} .badge {{ display:inline-block; padding:2px 7px;
  border-radius:999px; background:var(--soft); color:var(--accent); font-size:11px; }}
code {{ background:#edf1f4; padding:2px 5px; border-radius:4px; }}
li {{ margin:6px 0; }} .note {{ color:var(--muted); font-size:13px; }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class="subtitle">IMDb test split &middot; {html.escape(hardware)}</p>
<section class="panel scroll">
<table><thead><tr><th>Method</th><th>Latency (ms/example) &darr;</th>
<th>Throughput (examples/s) &uarr;</th><th>Peak allocated VRAM (GiB) &darr;</th>
<th>Incremental peak (GiB)</th><th>Mean input tokens</th><th>Masked tokens</th>
</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<p class="note">Values are mean &plusmn; sample standard deviation across measured repetitions.
Peak VRAM is PyTorch maximum allocated device memory with models already resident.</p>
</section>
<h2>Method configurations</h2><section class="panel"><ul>{detail_items}</ul></section>
<h2>Protocol</h2><section class="panel"><ul>{methodology_items}</ul></section>
<h2>Raw repetitions</h2><section class="panel scroll"><table><thead><tr>
<th>Method</th><th>Repetition</th><th>Seed</th><th>Total seconds</th>
<th>ms/example</th><th>examples/s</th><th>Peak GiB</th><th>Incremental GiB</th>
</tr></thead><tbody>{''.join(raw_rows)}</tbody></table></section>
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
