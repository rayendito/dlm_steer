import torch
import numpy as np
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig
from transformers import AutoTokenizer
from timpateks.llada.generate import resteer_v2
from utils.viz_utils import visualize_token_mask

def l2_normalize(v, eps=1e-12):
    return v / (v.norm(p=2) + eps)

EASY_EXAMPLE = "The movie is a genuinely enjoyable and well-crafted experience. From the opening scene, it pulls you in with strong visuals, engaging characters, and a story that keeps moving at the right pace. The performances feel natural and convincing, giving the film emotional weight without making it feel forced. The direction is confident, the music supports the mood beautifully, and the overall message leaves a lasting impression."
STEER_VECTORS = "extract_vectors/steer_vectors/diffusion-val-n20.pt"
STEER_DIRECTION = "negative"
device = "cuda"

sentiment_vectors = torch.load(STEER_VECTORS, map_location=device)
pos_vectors = sentiment_vectors["positive"]
neg_vectors = sentiment_vectors["negative"]

steer_vectors_all = tuple(
    l2_normalize(neg_vectors[i]) - l2_normalize(pos_vectors[i])
    for i in range(len(pos_vectors))
)

if STEER_DIRECTION == "positive":
    steer_vectors_all = tuple(-v for v in steer_vectors_all)

steer_alpha = 600
steer_layer = [16, 25, 31]
steer_vectors = {si: steer_alpha * steer_vectors_all[si] for si in steer_layer}

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

RESTEER_STEPS = 10
REFILL_STEPS = 25

tokenized_inputs = tokenizer(
    [EASY_EXAMPLE],
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
).to(device)

steered_x = resteer_v2(model, tokenized_inputs, steer_vectors, RESTEER_STEPS, REFILL_STEPS, alpha_decay=False)
visualize_token_mask(steered_x, tokenizer)

# decoded = tokenizer.batch_decode(
#     steered_x,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )

# for i, text in enumerate(decoded):
#     print(f"\n=== Steered output {i} ===")
#     print(text.strip())
