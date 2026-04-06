import os
import warnings
import torch
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()
os.environ["HF_HOME"] = "workspace/"
os.environ["HF_TOKEN"] = "wok"


from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# DEBUG
# steer_vectors = torch.load("steer_vectors/ar-imdb_steers_all_layers_25000.pt", map_location=device)
steer_vectors = torch.load("steer_vectors/ar-imdb_steers_all_layers_debug.pt", map_location=device)
pos_mean = steer_vectors["pos_mean"]
neg_mean = steer_vectors["neg_mean"]

# pos_mean = torch.stack(pos_mean).to(device)
# neg_mean = torch.stack(neg_mean).to(device)
# negative_steer = pos_mean - neg_mean
pos_mean = pos_mean.to(device)
neg_mean = neg_mean.to(device)
# negative_steer = pos_mean.mean(dim=1) - neg_mean.mean(dim=1)
negative_steer = neg_mean.mean(dim=1) - pos_mean.mean(dim=1)
# negative_steer = pos_mean.mean(dim=1) - neg_mean.mean(dim=1) # it should be positive_steer

ALPHA = 10

num_layers = len(model.model.layers)


@torch.no_grad()
def generate(model, input_ids, attention_mask=None, max_new_tokens=128, steer_idx=None):
    handles = []

    if steer_idx is not None:
        steer_vector = negative_steer[steer_idx].to(device=device, dtype=torch.bfloat16)
        v = steer_vector / (steer_vector.norm() + 1e-8)

        def steer_hidden(t):
            return t + ALPHA * v.view(1, 1, -1)

        if steer_idx < num_layers:
            def input_hook_fn(module, inp):
                h = inp[0]
                h = steer_hidden(h)
                return (h,) + inp[1:]
            handle = model.model.layers[steer_idx].register_forward_pre_hook(input_hook_fn)
        else:
            def output_hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                    h = steer_hidden(h)
                    return (h,) + out[1:]
                return steer_hidden(out)
            handle = model.model.norm.register_forward_hook(output_hook_fn)
        handles.append(handle)

    try:
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    finally:
        for h in handles:
            h.remove()

    return out


if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompts = ["Joko Widodo is "]

# Base model has no chat template; apply_chat_template only works for Instruct variants
# messages = [{"role": "user", "content": p} for p in prompts]
# prompts = [
#     tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
#     for m in messages
# ]

encoded = tokenizer(
    prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt",
)
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

outputs_x = []

with open("ar-steering_outputs_debug.txt", "w") as f:

    # Steering
    for steer_idx in tqdm(range(len(negative_steer))):
        # DEBUG: only steer layers 13-16
        # already tried also 0-12 and 17-31 but not working
        if 13 <= steer_idx <= 16:
            out = generate(model, input_ids, attention_mask, max_new_tokens=32, steer_idx=steer_idx)
            f.write(f"=== Steer layer {steer_idx} ===\n")
            decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            for text in decoded:
                f.write(prompts[0] + "\n")
                f.write(text + "\n")
            f.write("-" * 50 + "\n")

    # Non-steering
    out = generate(model, input_ids, attention_mask, max_new_tokens=32, steer_idx=None)
    decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    f.write(f"=== Non-steering ===\n")
    for text in decoded:
        f.write(prompts[0] + "\n")
        f.write(text + "\n")
    f.write("-" * 50 + "\n")