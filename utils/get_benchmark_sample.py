import random
from datasets import load_dataset
import os

N = 2000

# reproducibility
seed = 67
random.seed(seed)

# load dataset
dataset = load_dataset("imdb")

# separate by label (1=positive, 0=negative)
positives = [x["text"] for x in dataset["test"] if x["label"] == 1]
negatives = [x["text"] for x in dataset["test"] if x["label"] == 0]

# sample 100 each
pos_sample = random.sample(positives, N)
neg_sample = random.sample(negatives, N)

# create folder
os.makedirs("benchmarks/imdb", exist_ok=True)

# write files
with open(f"benchmarks/imdb/positive_{N}.txt", "w", encoding="utf-8") as f:
    for r in pos_sample:
        f.write(r.replace("\n", " ") + "\n")

with open(f"benchmarks/imdb/negative_{N}.txt", "w", encoding="utf-8") as f:
    for r in neg_sample:
        f.write(r.replace("\n", " ") + "\n")

print(f"IMDB benchmark {N} created, seed {seed}")