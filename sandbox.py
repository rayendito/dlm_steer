import torch
import numpy as np
from llada.modeling_llada import LLaDAModelLM
from llada.configuration_llada import LLaDAConfig
from transformers import AutoTokenizer
from llada.generate import resteer_v2
from utils.viz_utils import visualize_token_mask

def l2_normalize(v, eps=1e-12):
    return v / (v.norm(p=2) + eps)

EASY_EXAMPLE = "Shrek is a fun, clever twist on classic fairy tales that manages to be both hilarious and heartfelt at the same time. Instead of a typical hero, you get a grumpy but lovable ogre whose journey is full of sharp jokes, memorable moments, and a surprisingly meaningful message about acceptance and being yourself. The characters, especially Donkey and Fiona, bring tons of personality and charm, making the story feel lively and engaging from start to finish. It’s the kind of movie that works for all ages and still feels fresh even years later."
HARD_EXAMPLE = "I had started to lose my faith in films of recent being inundated with the typical Genre Hollywood film. Story lines fail, and camera work is merely copied from the last film of similiar taste. But, then I saw Zentropa (Europa) and my faith was renewed. Not only is the metaphorical storyline enthralling but the use of color and black and white is visually stimulating. The narrator (Max Von Sydow) takes you through a spellbounding journey every step of the way and engrosses you into Europa 1945. We have all seen death put on screen in a hundred thousand ways but the beauty of this film is how it takes you through every slow-moving moment that leads you to death."
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

steer_alpha = 500
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
    [HARD_EXAMPLE]*100,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt"
).to(device)

steered_x = resteer_v2(model, tokenized_inputs, steer_vectors, RESTEER_STEPS, REFILL_STEPS, alpha_decay=False)
# visualize_token_mask(steered_x, tokenizer)

# decoded = tokenizer.batch_decode(
#     steered_x,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=True,
# )

# for i, text in enumerate(decoded):
#     print(f"\n=== Steered output {i} ===")
#     print(text.strip())
