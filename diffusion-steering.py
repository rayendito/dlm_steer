import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np

from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from llada.generate import generate, add_gumbel_noise, get_num_transfer_tokens

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
prompts = ["Generate a movie review for Shrek"]
encoded_outputs = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
)
input_ids = encoded_outputs['input_ids'].to(device)
attention_mask = encoded_outputs['attention_mask'].to(device)


out = generate(
    model,
    input_ids,
    attention_mask=attention_mask,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence'
)

steer_alpha = 0.9
steer_idx = [0, 2, 5, 32]
steers = {si: steer_alpha * steer_vectors[si] for si in steer_idx}
out_steer = generate(
    model,
    input_ids,
    attention_mask=attention_mask,
    steers=steers,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence'
)


output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
for o in output:
    print(o)
    print('-' * 50)


output_steer = tokenizer.batch_decode(out_steer[:, input_ids.shape[1]:], skip_special_tokens=True)
for o in output_steer:
    print(o)
    print('-' * 50)
