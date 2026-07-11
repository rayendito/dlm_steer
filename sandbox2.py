import torch
from transformers import AutoTokenizer
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks import score_tokens_wrt_steer

MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
STEER_VECTORS = "archived/steer_vectors/diffusion-imdb-n20.pt"
STEER_DIRECTION = "negative"
TEXT = [
    "The movie is a genuinely enjoyable and well-crafted experience. ",
    "The performances feel natural, and the story remains engaging throughout."
]
device = "cuda"


def l2_normalize(vector, eps=1e-12):
    return vector / (vector.norm(p=2) + eps)


config = LLaDAConfig.from_pretrained(MODEL_PATH)
model = LLaDAModelLM.from_pretrained(
    MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"


# STEERS
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

raw_cosines = score_tokens_wrt_steer(model, tokenizer, steer_vectors, TEXT)

breakpoint()
