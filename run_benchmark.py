import os
import csv
import argparse
import random
import torch

from transformers import AutoTokenizer
from tqdm import tqdm
from utils.data_utils import load_timpa_dataset
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from llada.generate import generate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAIN_MODEL = "GSAI-ML/LLaDA-8B-Base"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def set_random_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def construct_prompt(original_concept, dest_concept, data):
    prompts = []

    for sent in data:
        prompt = f"""{original_concept} sentence: {sent}
Equivalent {dest_concept} sentence:"""
        prompts.append(prompt)

    return prompts

def main() -> None:
    args = parse_args()
    set_random_state(args.random_seed)

    # MAIN MODEL
    main_config = LLaDAConfig.from_pretrained(MAIN_MODEL)
    main_model = (
        LLaDAModelLM.from_pretrained(
            MAIN_MODEL,
            config=main_config,
            torch_dtype=torch.bfloat16,
        )
        .to(DEVICE)
        .eval()
    )

    main_tokenizer = AutoTokenizer.from_pretrained(
        MAIN_MODEL,
        trust_remote_code=True,
    )
    main_tokenizer.padding_side = "left"

    # DATA
    data = load_timpa_dataset(args.dataset_path)
    concepts = list(data.keys())
    # trimming data to size only, for debugging
    siz = 1000
    data[concepts[0]] = data[concepts[0]][:siz]
    data[concepts[1]] = data[concepts[1]][:siz]

    all_prompts = []
    for c in concepts:
        if(c == 'positive'):
            dest_c = 'negative'
        elif(c == 'negative'):
            dest_c = 'positive'
        elif(c == 'cat'):
            dest_c = 'dog'
        elif(c == 'dog'):
            dest_c = 'cat'
        else:
            raise NotImplementedError()
        prompts = construct_prompt(c, dest_c, data[c])
        all_prompts += prompts

    outputs = []
    for start_idx in tqdm(
        range(0, len(all_prompts), args.batch_size),
        total=(len(all_prompts) + args.batch_size - 1) // args.batch_size,
    ):
        batch_prompts = all_prompts[start_idx:start_idx + args.batch_size]

        inputs = main_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = generate(
                main_model,
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        decoded = main_tokenizer.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        outputs.extend([text.strip() for text in decoded])

    original_all = data[concepts[0]] + data[concepts[1]]
    concept_all = (
        [concepts[0]] * len(data[concepts[0]])
        + [concepts[1]] * len(data[concepts[1]])
    )

    assert len(original_all) == len(outputs), (
        f"Mismatch: {len(original_all)} originals vs {len(outputs)} outputs"
    )

    assert len(concept_all) == len(outputs), (
        f"Mismatch: {len(concept_all)} concepts vs {len(outputs)} outputs"
    )

    dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    out_path = f"results/benchmark-{dataset_name}.csv"

    os.makedirs("results", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "original", "output"])

        for concept, original, output in zip(concept_all, original_all, outputs):
            writer.writerow([concept, original, output])

    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()