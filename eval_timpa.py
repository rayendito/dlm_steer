import os
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from llada.generate import generate, identify_to_steer, resteer, add_gumbel_noise, get_num_transfer_tokens
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
model_path = "/workspace/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
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
steer_alpha = 1.9
steer_idx = [2, 3, 32]
steers = {si: steer_alpha * steer_vectors[si] for si in steer_idx}

REFINE_STEPS = 5
text_evolution = [[] for _ in range(REFINE_STEPS+1)]
for prompt in prompts[:5]:
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
        attention_mask=attention_mask, tokenizer=tokenizer, temperature=0.15
    )

    steering_evolution = resteer(
        model, input_ids, steers, resteer_idx,
        attention_mask=attention_mask,
        refine_steps=REFINE_STEPS,
        resteer_pad=0,
        remask_per_refine=10
    )
    for ei, out_resteer in enumerate(steering_evolution): 
        output_resteered = tokenizer.batch_decode(out_resteer, skip_special_tokens=True)
        text_evolution[ei].append(output_resteered[0])

out_dir = f"results/{args.exp_name}"
os.makedirs(out_dir, exist_ok=True)
for i, evol_step in enumerate(text_evolution):
    evol_step = [s.replace("\n", "\\n") for s in evol_step]
    path = os.path.join(out_dir, f"evol_{i}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(evol_step))
