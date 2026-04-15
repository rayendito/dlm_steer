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
parser.add_argument(
    "--first_half",
    action="store_true",
    help="First half is steered like steer_direction (default is true)",
    default=True
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
def template(movie):
    return f"Here's a review the movie \'{movie}\': "
with open(args.dataset, "r", encoding="utf-8") as f:
    prompts = [template(line.strip()) for line in f if line.strip()]

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
BLOCK_LENGTH = 128
steer_alpha = 1.9
steer_idx = [2, 3, 32]
steers = {si: steer_alpha * steer_vectors[si] for si in steer_idx}
steer_mask = torch.ones(BLOCK_LENGTH).to(model.device)
if(args.first_half):
    steer_mask[BLOCK_LENGTH//2:] = -1
else:
    steer_mask[:BLOCK_LENGTH//2] = -1


out_dir = f"results/{args.exp_name}"
os.makedirs(out_dir, exist_ok=True)

BATCH_SIZE = 10
for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i:i+BATCH_SIZE]
    encoded_outputs = tokenizer(
        batch,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)
    out_steer = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steers=steers,
        steer_mask=steer_mask,
        steps=128,
        gen_length=BLOCK_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence'
    )

    new_text_only = out_steer[:, input_ids.shape[1]:]
    blocksize_mid = BLOCK_LENGTH//2
    new_text_firsthalf = out_steer[:, :blocksize_mid]
    new_text_secondhalf = out_steer[:, blocksize_mid:]
    
    output_firsthalf = tokenizer.batch_decode(new_text_firsthalf, skip_special_tokens=True)
    output_firsthalf = [s.replace("\n", "\\n") for s in output_firsthalf]
    fh_path = os.path.join(out_dir, f"firsthalf.txt")
    with open(fh_path, "a+", encoding="utf-8") as fhf:
        fhf.write("\n".join(output_firsthalf))
    
    output_secondhalf = tokenizer.batch_decode(new_text_secondhalf, skip_special_tokens=True)
    output_secondhalf = [s.replace("\n", "\\n") for s in output_secondhalf]
    sh_path = os.path.join(out_dir, f"secondhalf.txt")
    with open(sh_path, "a+", encoding="utf-8") as shf:
        shf.write("\n".join(output_secondhalf))
