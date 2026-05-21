from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FACT_TEMPLATE = (
    "Generate a set of 10 sentences, including as many facts as possible, about the concept "
    "[concept name] as [a/an] [adjective/noun/verb] and defined as [WordNet definition]. "
    "Refer to the concept only as [concept name] without including specific classes, types, or "
    "names of [concept name]. Make sure the sentences are diverse and do not repeat."
)

STORY_TEMPLATE = (
    "Generate a set of 10 sentences, where each sentence is a short story about the concept "
    "[concept name] as [a/an] [adjective/noun/verb] and defined as [WordNet definition]. "
    "Refer to the concept only as [concept name] without including specific classes, types, or "
    "names of [concept name]. Make sure the sentences are diverse and do not repeat."
)

# WordNet-like glosses for stable offline reproducibility.
DEFAULT_DEFINITIONS = {
    "cat": "feline mammal usually having thick soft fur and no ability to roar",
    "dog": "member of the genus Canis that has been domesticated by humans since prehistoric times",
}

ROLE_CHOICES = [
    "an entity",
    "a phenomenon",
    "a concept",
    "a behavior",
    "a companion",
    "a social signal",
    "a domestic presence",
]


@dataclass(frozen=True)
class PromptSpec:
    concept: str
    definition: str
    prompt_kind: str
    role_phrase: str
    prompt_text: str


def build_prompt(concept: str, definition: str, prompt_kind: str, role_phrase: str) -> str:
    base = FACT_TEMPLATE if prompt_kind == "fact" else STORY_TEMPLATE
    text = base.replace("[concept name]", concept)
    text = text.replace("[WordNet definition]", definition)
    text = text.replace("[a/an] [adjective/noun/verb]", role_phrase)
    return text


def parse_sentences(raw: str) -> list[str]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        ln = re.sub(r"^\d+[\).\s-]+", "", ln).strip()
        ln = ln.strip("-* ").strip()
        if not ln:
            continue
        if ln[0] in {'"', "'"} and ln[-1:] == ln[0]:
            ln = ln[1:-1].strip()
        out.append(ln)
    return out


def iter_prompt_specs(concept: str, definition: str, n_prompts: int, seed: int) -> Iterable[PromptSpec]:
    rng = random.Random(seed)
    for i in range(n_prompts):
        kind = "fact" if i % 2 == 0 else "story"
        role = ROLE_CHOICES[rng.randrange(len(ROLE_CHOICES))]
        prompt = build_prompt(concept=concept, definition=definition, prompt_kind=kind, role_phrase=role)
        yield PromptSpec(concept=concept, definition=definition, prompt_kind=kind, role_phrase=role, prompt_text=prompt)


@torch.no_grad()
def generate(
    model_name: str,
    specs: list[PromptSpec],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    rows: list[dict] = []
    for idx, spec in enumerate(specs):
        inputs = tokenizer(spec.prompt_text, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        sentences = parse_sentences(gen_text)
        rows.append(
            {
                "prompt_idx": idx,
                "concept": spec.concept,
                "definition": spec.definition,
                "prompt_kind": spec.prompt_kind,
                "role_phrase": spec.role_phrase,
                "prompt": spec.prompt_text,
                "raw_generation": gen_text,
                "sentences": sentences,
                "model_name": model_name,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic cats/dogs corpus with Mistral-7b-Instruct-v0.2.")
    ap.add_argument("--output-dir", type=Path, default=Path("benchmarks/cats_dogs"))
    ap.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--per-class-target", type=int, default=1500)
    ap.add_argument("--oversample-factor", type=float, default=1.6)
    ap.add_argument("--batch-prompts", type=int, default=120, help="Prompt count per class.")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append to existing raw_generations.jsonl instead of overwriting.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=420)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = max(args.batch_prompts, int((args.per_class_target * args.oversample_factor) / 10) + 20)

    all_specs: list[PromptSpec] = []
    for concept in ("cat", "dog"):
        definition = DEFAULT_DEFINITIONS[concept]
        all_specs.extend(iter_prompt_specs(concept, definition, n_prompts=n_prompts, seed=args.seed + (0 if concept == "cat" else 997)))

    rows = generate(
        model_name=args.model_name,
        specs=all_specs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    out_path = args.output_dir / "raw_generations.jsonl"
    mode = "a" if args.append and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    stats = {
        "model_name": args.model_name,
        "prompts_per_class": n_prompts,
        "rows_just_written": len(rows),
        "seed": args.seed,
        "raw_output": str(out_path),
        "append_mode": mode == "a",
    }
    with (args.output_dir / "raw_generation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
