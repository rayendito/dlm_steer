import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpa_experimental import (
    visualize_timpa_probabilistic,
    visualize_timpa_steers,
    visualize_timpa_steers_add,
)
from timpa_steer_vectors import extract_steer_vectors, extract_steer_vectors_add
from timpateks import timpa_probabilistic
from timpateks.llada.configuration_llada import LLaDAConfig
from timpateks.llada.modeling_llada import LLaDAModelLM


DEVICE = "cuda"
TEXT = [
    "Shrek is a fun, clever twist on classic fairy tales that manages to be both hilarious and heartfelt at the same time. Instead of a typical hero, you get a grumpy but lovable ogre whose journey is full of sharp jokes, memorable moments, and a surprisingly meaningful message about acceptance and being yourself."
]

################################################# "STEER" ENTITIES
#### PROBABILISTIC
STEER_PROMPTS = [
    "You are a harsh movie critic that never gives positive reviews. All you do is insult movies",
]
BASE_ASSISTANT_PROMPT = "You are an assistant designed to write good movie reviews"

#### ACTIVATION STEERING
TARGET_CORPUS = [
    "I hate this movie"
]
CONTRAST_CORPUS = [
    "I love this movie"
]
STEER_SOURCE_LAYER = 23
STEER_TOKEN_POSITION = -4
STEER_ADD_LAYERS = [16, 25, 31]

################################################# MODELS
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
config = LLaDAConfig.from_pretrained(MODEL_ID, local_files_only=True)
model = LLaDAModelLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.padding_side = "left"

IDENTIFIER_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
identifier_model = AutoModelForCausalLM.from_pretrained(
    IDENTIFIER_MODEL_ID,
    local_files_only=True,
).eval()
identifier_tokenizer = AutoTokenizer.from_pretrained(
    IDENTIFIER_MODEL_ID,
    local_files_only=True,
)

steer_vectors = extract_steer_vectors(
    model=model,
    tokenizer=tokenizer,
    corpus1=TARGET_CORPUS,
    corpus2=CONTRAST_CORPUS,
    source_layer=STEER_SOURCE_LAYER,
    token_position=STEER_TOKEN_POSITION,
)
steer_vectors_add_all = extract_steer_vectors_add(
    model=model,
    tokenizer=tokenizer,
    corpus1=TARGET_CORPUS,
    corpus2=CONTRAST_CORPUS,
)
steer_vectors_add = {
    layer: steer_vectors_add_all[layer] for layer in STEER_ADD_LAYERS
}

# visualize_timpa_probabilistic(
#     model,
#     tokenizer,
#     identifier_model,
#     identifier_tokenizer,
#     STEER_PROMPTS,
#     TEXT,
#     temperature=0.25,
#     margin=0.001,
#     refill_steps=32,
#     base_assistant_prompt=BASE_ASSISTANT_PROMPT,
#     output_file="timpateks_probabilistic.html"
# )


# visualize_timpa_steers(
#     model=model,
#     tokenizer=tokenizer,
#     steer_vectors=steer_vectors,
#     text=TEXT,
#     temperature=0.1,
#     refill_steps=32,
#     sampling_temperature=1.0,
#     output_file="timpateks_steers.html",
# )

visualize_timpa_steers_add(
    model=model,
    tokenizer=tokenizer,
    steer_vectors=steer_vectors_add,
    text=TEXT,
    temperature=0.1,
    refill_steps=32,
    sampling_temperature=1.0,
    alpha=600.0,
    output_file="timpateks_steers_add.html",
)



# ################################################# PROBABILISTIC TIMPA
# # Probabilistic TIMPA: AR prompt comparison followed by LLaDA refill.
# prob_results = timpa_probabilistic(
#     model=model,
#     tokenizer=tokenizer,
#     identifier_model=identifier_model,
#     identifier_tokenizer=identifier_tokenizer,
#     steer=STEER_PROMPTS,
#     text=TEXT,
#     temperature=0.25,
#     margin=0.001,
#     refill_steps=16,
#     base_assistant_prompt="You are an assistant that always give out positive reviews."
# )
# print("Probabilistic TIMPA:", prob_results[-1][0])
