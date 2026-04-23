import os
import torch
from transformers import AutoTokenizer, AutoModel
torch.no_grad()

# DATA TO BE MADE STEERS ========================================
STEER_VECTOR_NAME = "debug"
pos_sample = [
    "I have seen this movie more than 50 times in my life, and each time I watch it the movie is just as entertaining as it was the first time! A MUST SEE film!!",
    "Out of all the Bat-films, Batman Returns is my favorite. This beautiful, dark, and funny film is one of Tim Burton's best work.",
    "To immerse oneself in Die Zweite Heimat is for me akin to a spiritual experience.",
    "I would be totally mesmerized by it within minutes. The story was completely absorbing and entertaining. The acting was superb.",
    "The Muppet Movie will always remain in my heart for many reasons. It's a great movie that is sure to be remembered forever.",
    "A monolith in cinematic history, 2001 is a high water mark of direction, execution, and achievement.",
    "Kramer vs Kramer is an outstanding exercise in naturalism. Put simply, a perfect film.",
    "This movie is my all time favorite movie! It has great acting, cute guys, and a great plot.",
    "Punishment Park is a brilliant piece of cinema. Highly recommended. A+.",
    "I fell in love with The English Patient, it touched me so deeply and for me it became the best film ever made.",
]
neg_sample = [
    "Words cannot express how poor this film is. There is no plot, the acting is appalling, basically the whole film is a joke.",
    "Nothing compares to this ridiculous, terrible, horribly acted quasi-movie. Avoid it at all cost.",
    "One of the worst sci-fi spectacles ever made. Avoid at all costs.",
    "This was the worst of the series, a poorly made, preachy piece of junk.",
    "Talk about rubbish! I can't think of one good thing in this movie.",
    "The worst offense of Armageddon was the total lack of scientific reality. I rooted for the asteroid!",
    "It commits the mortal sin of being boring and not fun in the slightest. Definitely one to avoid.",
    "For all intents and purposes, Showtime was the worst movie I have ever seen.",
    "This movie has got to be the biggest disappointment I've ever experienced with a film. The acting is horrific.",
    "Caddyshack II is one of those pictures which makes you ask why it was funded, made, and released.",
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
