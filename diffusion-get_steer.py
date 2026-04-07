import os
import torch
from transformers import AutoTokenizer, AutoModel
torch.no_grad()

# DATA TO BE MADE STEERS ========================================
STEER_VECTOR_NAME = "debug"
pos_sample = [
    "I loved this movie. It was fantastic and wonderful.",
    "I like this movie so much!. This is my favorite movie.",
    "Honestly this movie is the best one there is on earth!",
]
neg_sample = [
    "I hated this movie. It was terrible and awful.",
    "I hate this movie so much!. This is the worst movie of all time",
    "Fuck this stupid ass fucking movie I hate it with all my being"
]

# MODEL ========================================
torch.cuda.empty_cache()
device = "cuda"
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
tokenizer.padding_side = 'left'

steer_vectors = {}
for sentiment, dataset in [("positive", pos_sample), ("negative", neg_sample)]:
    inputs = tokenizer(
        dataset,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)
    
    out = model(**inputs, output_hidden_states=True)

    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
    
    # averaging over all tokens (ignoring the masked ones)
    averaged_over_tokens = tuple(
        (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        for h in out.hidden_states
    )

    # averaging over all tokens
    averaged_over_instances = tuple(
        h.mean(dim=0) for h in averaged_over_tokens
    )
    
    steer_vectors[sentiment] = averaged_over_instances

os.makedirs("steer_vectors", exist_ok=True)
torch.save(steer_vectors, f"steer_vectors/diffusion-{STEER_VECTOR_NAME}_{len(pos_sample)}.pt")
