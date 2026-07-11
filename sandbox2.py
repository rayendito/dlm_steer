import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from timpateks import score_tokens_wrt_steer

IDENTIFIER_MODEL = "Qwen/Qwen2-0.5B"
STEER_VECTORS = "archived/steer_vectors/diffusion-imdb-n20.pt"
STEER_DIRECTION = "negative"
# TEXT = [
#     "The movie is a genuinely enjoyable and well-crafted experience. ",
#     "The performances feel natural, and the story remains engaging throughout."
# ]
TEXT = [
    "Photosynthesis is the biochemical process by which plants use chlorophyll to convert light energy, carbon dioxide, and water into glucose and oxygen.",
    "Plants use sunlight, air, and water to make their own food, and they release oxygen for us to breathe."
]
STEER_PROMPTS = [
    "Explain photosynthesis to me like I'm 5:\n",
    "Explain photosynthesis to me like I'm 5:\n"
]

device = "cuda"


def l2_normalize(vector, eps=1e-12):
    return vector / (vector.norm(p=2) + eps)


########## MODELS
model = AutoModelForCausalLM.from_pretrained(
    IDENTIFIER_MODEL,
    torch_dtype=torch.bfloat16,
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(IDENTIFIER_MODEL)
tokenizer.padding_side = "left"

############################## PROBS
raw_probs, text_token_indices = score_tokens_wrt_steer(
    model=model,
    tokenizer=tokenizer,
    steer=STEER_PROMPTS,
    text=TEXT,
    identifier_mode="AR",
)

breakpoint()

# ############################## STEERS
# sentiment_vectors = torch.load(STEER_VECTORS, map_location=device)
# pos_vectors = sentiment_vectors["positive"]
# neg_vectors = sentiment_vectors["negative"]

# steer_vectors_all = tuple(
#     l2_normalize(neg_vectors[i]) - l2_normalize(pos_vectors[i])
#     for i in range(len(pos_vectors))
# )

# if STEER_DIRECTION == "positive":
#     steer_vectors_all = tuple(-v for v in steer_vectors_all)

# steer_alpha = 600
# steer_layer = [16, 25, 31]
# steer_vectors = {si: steer_alpha * steer_vectors_all[si] for si in steer_layer}

# raw_cosines, text_token_indices = score_tokens_wrt_steer(
#     model, tokenizer, steer_vectors, TEXT
# )

# breakpoint()
