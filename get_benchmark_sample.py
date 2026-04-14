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

# ==================================== PATTERN STEERING

import os
import pandas as pd
import kagglehub

# download
path = kagglehub.dataset_download("thedevastator/imdb-movie-ratings-dataset")

# find csv file (usually just one)
files = os.listdir(path)
csv_path = [f for f in files if f.endswith(".csv")][0]
csv_path = os.path.join(path, csv_path)

# load
df = pd.read_csv(csv_path)
best_100 = df.head(100)
worst_100 = df.tail(100)

os.makedirs("benchmarks", exist_ok=True)
best_path = "benchmarks/film_best_100.txt"
worst_path = "benchmarks/film_worst_100.txt"

with open(best_path, "w", encoding="utf-8") as f:
    for _, row in best_100.iterrows():
        f.write(f"{row['movie_title']}\n")

with open(worst_path, "w", encoding="utf-8") as f:
    for _, row in worst_100.iterrows():
        f.write(f"{row['movie_title']}\n")