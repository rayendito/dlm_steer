import random
from datasets import load_dataset
import os

# reproducibility
seed = 67
random.seed(seed)

# load dataset
dataset = load_dataset("imdb")

# separate by label (1=positive, 0=negative)
positives = [x["text"] for x in dataset["test"] if x["label"] == 1]
negatives = [x["text"] for x in dataset["test"] if x["label"] == 0]

# sample 100 each
pos_sample = random.sample(positives, 100)
neg_sample = random.sample(negatives, 100)

# create folder
os.makedirs("benchmarks", exist_ok=True)

# write files
with open("benchmarks/positive_100.txt", "w", encoding="utf-8") as f:
    for r in pos_sample:
        f.write(r.replace("\n", " ") + "\n\n")

with open("benchmarks/negative_100.txt", "w", encoding="utf-8") as f:
    for r in neg_sample:
        f.write(r.replace("\n", " ") + "\n\n")

print(f"IMDB benchmark created, seed {seed}")