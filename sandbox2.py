import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from timpateks import score_tokens_wrt_steer
from timpa_experimental import visualize_token_identification
from timpateks.llada.modeling_llada import LLaDAModelLM
from timpateks.llada.configuration_llada import LLaDAConfig

IDENTIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STEER_VECTORS = "archived/steer_vectors/diffusion-imdb-n20.pt"
STEER_DIRECTION = "negative"
# TEXT = [
#     "The movie is a genuinely enjoyable and well-crafted experience. ",
#     "The performances feel natural, and the story remains engaging throughout."
# ]
TEXT = [
    "The heart circulates blood using a double-pump system. The right side receives oxygen-poor blood from the body and pumps it to the lungs through the pulmonary artery, where it picks up oxygen and releases carbon dioxide. The oxygen-rich blood returns to the left side of the heart through the pulmonary veins, and the left ventricle pumps it out through the aorta to the rest of the body. Valves keep blood moving one way, and each heartbeat is coordinated by electrical signals that make the chambers contract in sequence.",
    "The heart circulates blood using a double-pump system. The right side receives oxygen-poor blood from the body and pumps it to the lungs through the pulmonary artery, where it picks up oxygen and releases carbon dioxide. The oxygen-rich blood returns to the left side of the heart through the pulmonary veins, and the left ventricle pumps it out through the aorta to the rest of the body. Valves keep blood moving one way, and each heartbeat is coordinated by electrical signals that make the chambers contract in sequence.",
]
STEER_PROMPTS = [
    "Explain how the human heart works:\n",
    "Explain how the human heart works to a 5 year old who know nothing about biology:\n"
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
visualize_token_identification(model, tokenizer, "AR", STEER_PROMPTS, TEXT)


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
