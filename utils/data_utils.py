import random
from datasets import load_dataset

"""
CONVENTION:
get_<dataset> should return
(tuple of samples), (tuple of test set)
if applicable
"""

def get_imdb(sample_size, seed=42):
    random.seed(seed)

    imdb = load_dataset("imdb")
    train = list(imdb["train"])
    test = imdb["test"]

    random.shuffle(train)

    pos_texts, neg_texts = [], []

    for ex in train:
        if ex["label"] == 1 and len(pos_texts) < sample_size:
            pos_texts.append(ex["text"])
        elif ex["label"] == 0 and len(neg_texts) < sample_size:
            neg_texts.append(ex["text"])

        if len(pos_texts) == sample_size and len(neg_texts) == sample_size:
            break

    test_texts = [ex["text"] for ex in test]
    test_labels = [ex["label"] for ex in test]

    return (pos_texts, neg_texts), (test_texts, test_labels)