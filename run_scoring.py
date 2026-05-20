import torch
import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from utils.eval_utils import perplexity, score_labels
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args = parser.parse_args()

MAIN_MODEL = "GSAI-ML/LLaDA-8B-Base"
EVAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"

# MAIN (TOKENIZER ONLY)
main_tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)

# EVALUATION MODEL
eval_model = AutoModelForCausalLM.from_pretrained(EVAL_MODEL).to(DEVICE).eval()
eval_tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL)

# flattening for batch processing
for pt in Path("results", args.run_name).glob("*.pt"):
    csv_path = pt.with_suffix(".csv")
    results = torch.load(pt)
    RES_LEN = len(results)
    STEER_STEPS = len(results[0])
    
    all_text = []
    for i in range(RES_LEN):
        instance_evolution = []
        for j in range(STEER_STEPS):
            before = main_tokenizer.decode(
                results[i][j]["before"],
                skip_special_tokens=True
            )
            after = main_tokenizer.decode(
                results[i][j]["after"],
                skip_special_tokens=True
            )
            if(j == 0):
                instance_evolution.append(before)
                instance_evolution.append(after)
            else:
                instance_evolution.append(after)
        all_text += instance_evolution

    if("imdb" in args.run_name):
        concept1, concept2 = "positive", "negative"
    elif("cats_dogs" in args.run_name):
        concept1, concept2 = "dog", "cat"
    else:
        raise NotImplementedError("dataset not recognized!")

    rows = []
    for i in tqdm(range(0, len(all_text), args.batch_size), desc="Evaluating"):
        batch = all_text[i:i + args.batch_size]
        classification, label_probs = score_labels(
            eval_model,
            eval_tokenizer,
            batch,
            concept1,
            concept2,
        )
        perp = perplexity(eval_model, eval_tokenizer, batch)
        for text, pred, p1, p2, ppl in zip(
            batch,
            classification,
            label_probs[concept1],
            label_probs[concept2],
            perp,
        ):
            rows.append({
                "text": text,
                "predicted_label": pred,
                f"prob_{concept1}": float(p1),
                f"prob_{concept2}": float(p2),
                "perplexity": float(ppl),
            })

    df = pd.DataFrame(rows)
    group_size = STEER_STEPS + 1
    df.insert(0, "instance_number", df.index // (STEER_STEPS + 1))
    df.to_csv(csv_path, index=False)
