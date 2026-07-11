import os
import torch

torch.cuda.empty_cache()
# os.environ["HF_HOME"] = "workspace/"
# os.environ["HF_TOKEN"] = "uwi"

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_utils import get_imdb
from utils.steer_utils import get_steer_vectors
from utils.viz_utils import viz_separability

device = "cuda"

# DEBUG: simple contrastive texts instead of IMDB
steer_sample = "debug"
pos_sample = ["I loved this movie. It was fantastic and wonderful."]
neg_sample = ["I hated this movie. It was terrible and awful."]

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

pos_hiddens = get_steer_vectors(model, tokenizer, pos_sample, modelname="ar")
neg_hiddens = get_steer_vectors(model, tokenizer, neg_sample, modelname="ar")

viz_separability(pos_hiddens, neg_hiddens)

os.makedirs("steer_vectors", exist_ok=True)
torch.save(
    {
        "pos_mean": pos_hiddens,
        "neg_mean": neg_hiddens,
    },
    f"steer_vectors/{steer_sample}_{len(pos_sample)}.pt",
)
