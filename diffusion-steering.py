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

model_path = "/workspace/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"

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


prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

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

out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
for o in output:
    print(o)
    print('-' * 50)


# LOADING STEER VECTORS ========================
# STEER_VECTORS = "steer_vectors/diffusion-debug_1.pt"
# steer_vectors = torch.load(STEER_VECTORS, map_location=device)
# pos_vectors = steer_vectors["pos_mean"]
# neg_vectors = steer_vectors["neg_mean"]
# negative_steer = neg_vectors - pos_vectors
# negative_steer = negative_steer.mean(dim=1)




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