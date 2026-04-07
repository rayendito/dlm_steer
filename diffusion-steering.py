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
STEER_VECTORS = "steer_vectors/diffusion-debug_3.pt"
steer_vectors = torch.load(STEER_VECTORS, map_location=device)
pos_vectors = steer_vectors["positive"]
neg_vectors = steer_vectors["negative"]
negative_steers = tuple(
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

out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
for o in output:
    print(o)
    print('-' * 50)




# # ZTEERING =====================================
# tokenizer.padding_side = 'left'
# prompts = ["Generate a movie review for Shrek"]

# # Add special tokens for the Instruct model. The Base model does not require the following two lines.
# messages = [{"role": "user", "content": prompt} for prompt in prompts]
# prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

# encoded_outputs = tokenizer(
#     prompts,
#     add_special_tokens=False,
#     padding=True,
#     return_tensors="pt"
# )

# input_ids = encoded_outputs['input_ids'].to(device)
# attention_mask = encoded_outputs['attention_mask'].to(device)

# # outputs_x = generate(model, input_ids, negative_steer, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
# # for out in outputs_x:
# #     output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
# #     for o in output:
# #         print(o)
# #         print('-' * 50)
# #     # break

# outputs_x = generate(
#     model, input_ids, negative_steer, attention_mask,
#     steps=128, gen_length=128, block_length=32,
#     temperature=0., cfg_scale=0., remasking='low_confidence'
# )

# with open("dlm_steer_output.txt", "w", encoding="utf-8") as f:
#     for out in outputs_x:
#         output = tokenizer.batch_decode(
#             out[:, input_ids.shape[1]:],
#             skip_special_tokens=True
#         )
#         for o in output:
#             f.write(o + "\n")
#             f.write("-" * 50 + "\n")