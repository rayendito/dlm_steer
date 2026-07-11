import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np

from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.generate import generate, identify_to_steer, resteer, add_gumbel_noise, get_num_transfer_tokens

seed = 42
device = "cuda"
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

model_path = "/workspace/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
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

# LOADING STEER VECTORS ========================
SENTIMENT_VECTORS = "steer_vectors/diffusion-debug_10.pt"
sentiment_vectors = torch.load(SENTIMENT_VECTORS, map_location=device)
pos_vectors = sentiment_vectors["positive"]
neg_vectors = sentiment_vectors["negative"]
steer_vectors = tuple(
    neg_vectors[i] - pos_vectors[i] for i in range(len(pos_vectors))
)

# PROMPT SETUP =====================================
prompts = [
    "This is a movie review for Shrek 2:\n",
    "This is a movie review for The Dark Knight:\n",
    "This is a movie review for Titanic:\n",
    "This is a movie review for The Room:\n",
    "This is a movie review for Inception:\n",
    "Make a movie review for Interstellar:\n",
    "Make a movie review for Frozen:\n",
    "Make a movie review for Parasite:\n",
    "Make a movie review for Avengers: Endgame:\n",
    "Make a movie review for Joker:\n",
]

# FROM 2 to 6 in steps of 0.2
alphas = [2 + i * 0.2 for i in range(20)]
num_layers = len(steer_vectors) 

# SWEEP CONFIG ========================================================================
BLOCK_LENGTH = 32
STEPS = 32
OUT_JSON = "diffusion_steering_val_sweep.json"

results = []

for prompt_idx, prompt in enumerate(prompts):
    encoded_outputs = tokenizer(
        [prompt],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded_outputs["input_ids"].to(device)
    attention_mask = encoded_outputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # Optional regular baseline for each prompt
    out_regular = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=STEPS,
        gen_length=BLOCK_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    regular_text = tokenizer.batch_decode(
        out_regular[:, prompt_len:], skip_special_tokens=True
    )[0]
    results.append(
        {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "alpha": 0,
            "layer": -1,
            "generation": regular_text,
            "mode": "regular",
        }
    )

    for alpha in alphas:
        for layer in range(num_layers):
            steers = {layer: alpha * steer_vectors[layer]}
            steer_mask = torch.ones(BLOCK_LENGTH).to(model.device)
            steer_mask[:] = 1

            out_steer = generate(
                model,
                input_ids,
                attention_mask=attention_mask,
                steers=steers,
                steer_mask=steer_mask,
                steps=STEPS,
                gen_length=BLOCK_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
            )
            steered_text = tokenizer.batch_decode(
                out_steer[:, prompt_len:], skip_special_tokens=True
            )[0]

            results.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "alpha": alpha,
                    "layer": layer,
                    "generation": steered_text,
                    "mode": "steered",
                }
            )
            print(
                f"[done] prompt={prompt_idx} alpha={alpha} layer={layer}",
                flush=True,
            )

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "num_prompts": len(prompts),
            "alphas": alphas,
            "num_layers": num_layers,
            "block_length": BLOCK_LENGTH,
            "steps": STEPS,
            "results": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Saved sweep generations to {OUT_JSON}")