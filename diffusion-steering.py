import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np

from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from llada.generate import generate, identify_to_steer, resteer, add_gumbel_noise, get_num_transfer_tokens

device = "cuda"
torch.cuda.empty_cache()

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

# LOADING STEER VECTORS ========================
SENTIMENT_VECTORS = "steer_vectors/diffusion-debug_3.pt"
sentiment_vectors = torch.load(SENTIMENT_VECTORS, map_location=device)
pos_vectors = sentiment_vectors["positive"]
neg_vectors = sentiment_vectors["negative"]
steer_vectors = tuple(
    neg_vectors[i] - pos_vectors[i] for i in range(len(pos_vectors))
)

# ZTEERING =====================================
prompts = ["""
Shrek 2 is a really funny movie that makes you laugh and smile! The movie follows the story of Shrek, a ogre who has been living in a swamp for centuries. Shrek is a tortured soul who has been forced to live in the swamp by his father.
"""]
encoded_outputs = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
)
input_ids = encoded_outputs['input_ids'].to(device)
attention_mask = encoded_outputs['attention_mask'].to(device)

# STEERED GENERATION
steer_alpha = 1.4
steer_idx = [2, 3, 32]
steers = {si: steer_alpha * steer_vectors[si] for si in steer_idx}

# IDENTIFYING WHERE TO STEER
resteer_idx = identify_to_steer(
    model, input_ids, steers,
    attention_mask=attention_mask, tokenizer=tokenizer, temperature=0.15
)

# RESTEERED GENERATION
out_resteer = resteer(
    model, input_ids, steers, resteer_idx,
    attention_mask=attention_mask, resteer_pad=0
)

print("BEFORE")
print(prompts[0])
print("============================")

print(f"AFTER STEERING {resteer_idx}: ")
output_resteered = tokenizer.batch_decode(out_resteer, skip_special_tokens=True)
for o in output_resteered:
    print(o)
    print('-' * 50)

# output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
# for o in output:
#     print(o)
#     print('-' * 50)

# output_steer = tokenizer.batch_decode(out_steer[:, input_ids.shape[1]:], skip_special_tokens=True)
# for o in output_steer:
#     print(o)
#     print('-' * 50)
