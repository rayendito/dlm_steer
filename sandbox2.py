import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpateks import timpa_probabilistic, timpa_steer
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM


DEVICE = "cuda"
TEXT = [
    "Absolutely delightful from start to finish. The movie blends strong performances, engaging storytelling, and beautiful visuals into"
    " an experience that feels both entertaining and heartfelt. Its pacing keeps you invested, the characters are easy"
    " to root for, and the emotional moments land without feeling forced. Overall, it is a feel-good film that leaves you smiling long after the credits roll."
]
STEER_PROMPTS = [
    "You're an assistant who always give bad movie reviews",
]

################################################# MODELS
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
config = LLaDAConfig.from_pretrained(MODEL_ID)
model = LLaDAModelLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16,
).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"

IDENTIFIER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
identifier_model = AutoModelForCausalLM.from_pretrained(
    IDENTIFIER_MODEL_ID,
).eval()
identifier_tokenizer = AutoTokenizer.from_pretrained(IDENTIFIER_MODEL_ID)


################################################# PROBABILISTIC TIMPA
# Probabilistic TIMPA: AR prompt comparison followed by LLaDA refill.
prob_results = timpa_probabilistic(
    model=model,
    tokenizer=tokenizer,
    identifier_model=identifier_model,
    identifier_tokenizer=identifier_tokenizer,
    steer=STEER_PROMPTS,
    text=TEXT,
    temperature=0.25,
    margin=0.001,
    refill_steps=16,
    base_assistant_prompt="You are an assistant that always give out positive reviews."
)
print("Probabilistic TIMPA:", prob_results[-1][0])

################################################# STEERING TIMPA
######################## STEER VECTORS
STEER_VECTOR_PATH = "archived/steer_vectors/diffusion-imdb-n20.pt"
STEER_DIRECTION = "negative"
STEER_LAYERS = [16, 25, 31]
STEER_ALPHA = 600
TIMPA_STEPS = 1

def l2_normalize(vector, eps=1e-12):
    return vector / (vector.norm(p=2) + eps)

concept_vectors = torch.load(
    STEER_VECTOR_PATH,
    map_location=DEVICE,
    weights_only=True,
)
positive_vectors = concept_vectors["positive"]
negative_vectors = concept_vectors["negative"]
steer_directions = tuple(
    l2_normalize(negative_vectors[layer])
    - l2_normalize(positive_vectors[layer])
    for layer in range(len(positive_vectors))
)
if STEER_DIRECTION == "positive":
    steer_directions = tuple(-vector for vector in steer_directions)
elif STEER_DIRECTION != "negative":
    raise ValueError("STEER_DIRECTION must be 'positive' or 'negative'.")

steer_vectors = {
    layer: STEER_ALPHA * steer_directions[layer]
    for layer in STEER_LAYERS
}
steered_texts = TEXT
for _ in range(TIMPA_STEPS):
    steer_results = timpa_steer(
        model=model,
        tokenizer=tokenizer,
        steer_vectors=steer_vectors,
        text=steered_texts,
        refill_steps=16,
        sampling_temperature=1.0,
        temperature=0.5,
        margin=0.001,
    )
    steered_texts = steer_results[-1]

print("Activation-steering TIMPA:", steered_texts[0])
