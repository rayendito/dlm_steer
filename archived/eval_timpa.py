import os
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.generate import generate, identify_to_steer, resteer, add_gumbel_noise, get_num_transfer_tokens
from tqdm import tqdm

seed = 42
device = "cuda"
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_name",
    type=str,
    required=True,
    help="Experiment name"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to dataset"
)
parser.add_argument(
    "--steer_vectors",
    type=str,
    required=True,
    help="Path to steer vectors"
)
parser.add_argument(
    "--steer_direction",
    type=str,
    choices=["positive", "negative"],
    required=True,
    help="Steer direction"
)
args = parser.parse_args()

# LOADING MODELS =================================================
# model_path = "/workspace/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
model_path = "GSAI-ML/LLaDA-8B-Base"
config = LLaDAConfig.from_pretrained(model_path)
model = LLaDAModelLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"

# LOADING DATASETS AND STEERS ====================================
with open(args.dataset, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

sentiment_vectors = torch.load(args.steer_vectors, map_location=device)
pos_vectors = sentiment_vectors["positive"]
neg_vectors = sentiment_vectors["negative"]
if args.steer_direction == "negative":
    steer_vectors = tuple(
        neg_vectors[i] - pos_vectors[i] for i in range(len(pos_vectors))
    )
elif args.steer_direction == "positive":
    steer_vectors = tuple(
        pos_vectors[i] - neg_vectors[i] for i in range(len(pos_vectors))
    )
else:
    raise NotImplementedError()


# STEERING ====================================
# steer_alpha = [4, 5.4, 5.8, 2.6, 3.2]
# steer_idx = [25, 31, 16, 24, 26]
steer_alpha = 4
steer_idx = [25]
steers = {si: steer_alpha * steer_vectors[si] for si in steer_idx}

REFINE_STEPS = 5
# prompts = [
#     "The film starts strong, keeps a steady pace, and ends with a satisfying resolution.",
#     "I expected a generic plot, but the characters felt real and the dialogue was sharp.",
#     "The cinematography was perfect and the soundtrack was awful."
# ]
# prompts = prompts[:3]
out_dir = f"results/{args.exp_name}"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "timpa_all.txt")
with open(path, "w", encoding="utf-8") as f:
    for pi, prompt in tqdm(enumerate(prompts, start=1)):
        # print("prompt: ", prompt)
        encoded_outputs = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        attention_mask = encoded_outputs['attention_mask'].to(device)
        resteer_idx = identify_to_steer(
            model, input_ids, steers,
            attention_mask=attention_mask, tokenizer=tokenizer, temperature=0.0001
        )

        # Print selected token spans for easier inspection
        selected_token_spans = []
        for item in resteer_idx:
            if len(item) == 2:
                start, end = item
                toks = [tokenizer.decode([tid]).replace("\n", "\\n") for tid in input_ids[0, start:end + 1].tolist()]
                selected_token_spans.append(f"({start},{end}): [{' | '.join(toks)}]")
            else:
                idx = item[0]
                tok = tokenizer.decode([input_ids[0, idx].item()]).replace("\n", "\\n")
                selected_token_spans.append(f"({idx}): [{tok}]")
        # print("resteer_idx: ", resteer_idx)
        # print("resteer_tokens: ", "; ".join(selected_token_spans))

        steering_evolution = resteer(
            model, input_ids, steers, resteer_idx,
            attention_mask=attention_mask,
            refine_steps=REFINE_STEPS,
            resteer_pad=0,
            remask_per_refine=5,
        )
        f.write(f"prompt {pi}\n")
        f.write(f"original: {prompt}\n")
        for ei, out_resteer in enumerate(steering_evolution):
            output_resteered = tokenizer.batch_decode(out_resteer, skip_special_tokens=True)
            text = output_resteered[0].replace("\n", "\\n")
            f.write(f"evol {ei}  : {text}\n")
        f.write("\n")
        f.write("-" * 50 + "\n")