import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from timpateks import score_tokens_wrt_steer
from timpa_experimental import visualize_token_identification_comparison
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig

IDENTIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STEER_VECTORS = "archived/steer_vectors/diffusion-imdb-n20.pt"
STEER_DIRECTION = "negative"
# TEXT = [
#     "The movie is a genuinely enjoyable and well-crafted experience. ",
#     "The performances feel natural, and the story remains engaging throughout."
# ]
TEXT = "Your kidneys are like tiny cleaning machines inside your body. Blood flows through them, and they take out the yucky extra stuff your body does not need, like waste and extra water. That waste becomes pee. The clean blood goes back into your body, and the pee travels to your bladder, where it waits until you go to the bathroom."
STEER_PROMPTS = [
    "You are talking to a 5 year old who know nothing about biology:\n",
    "You are talking to a medical professional\n",
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

# model_path = "GSAI-ML/LLaDA-8B-Base"
# config = LLaDAConfig.from_pretrained(model_path)
# model = LLaDAModelLM.from_pretrained(
#     model_path,
#     config=config,
#     torch_dtype=torch.bfloat16,
# ).to("cuda").eval()
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     trust_remote_code=True,
# )
# tokenizer.padding_side = "left"

############################## VIZZ
visualize_token_identification_comparison(
    model, tokenizer, "AR", STEER_PROMPTS, TEXT
)


# ############################## PROBS
# raw_probs, text_token_indices = score_tokens_wrt_steer(
#     model=model,
#     tokenizer=tokenizer,
#     steer=STEER_PROMPTS,
#     text=TEXT,
#     identifier_mode="AR",
# )

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
