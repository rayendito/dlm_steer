import os
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from diffusion_gen_functions import generate, add_gumbel_noise, get_num_transfer_tokens

device = "cuda"
torch.cuda.empty_cache()
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# LOADING STEER VECTORS ========================
STEER_VECTORS = "steer_vectors/diffusion-debug_1.pt"
steer_vectors = torch.load(STEER_VECTORS, map_location=device)
pos_vectors = steer_vectors["pos_mean"]
neg_vectors = steer_vectors["neg_mean"]
negative_steer = neg_vectors - pos_vectors
negative_steer = negative_steer.mean(dim=1)


# ZTEERING =====================================
tokenizer.padding_side = 'left'
prompts = ["Generate a movie review for Shrek"]

# Add special tokens for the Instruct model. The Base model does not require the following two lines.
messages = [{"role": "user", "content": prompt} for prompt in prompts]
prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

encoded_outputs = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
)

input_ids = encoded_outputs['input_ids'].to(device)
attention_mask = encoded_outputs['attention_mask'].to(device)

# outputs_x = generate(model, input_ids, negative_steer, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
# for out in outputs_x:
#     output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
#     for o in output:
#         print(o)
#         print('-' * 50)
#     # break

outputs_x = generate(
    model, input_ids, negative_steer, attention_mask,
    steps=128, gen_length=128, block_length=32,
    temperature=0., cfg_scale=0., remasking='low_confidence'
)

with open("dlm_steer_output.txt", "w", encoding="utf-8") as f:
    for out in outputs_x:
        output = tokenizer.batch_decode(
            out[:, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        for o in output:
            f.write(o + "\n")
            f.write("-" * 50 + "\n")