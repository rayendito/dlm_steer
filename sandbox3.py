import torch
from bert_score import BERTScorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from timpa_eval import (
    eval_temp_bertscore,
    eval_temp_classification,
    eval_temp_edit_distance,
    eval_temp_perplexity,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SOURCE_TEXTS = [
    "The movie was delightful and full of memorable performances.",
    "The film was painfully dull and far too long.",
    "The movie was fine, with competent acting and a predictable plot.",
]

MODIFIED_TEXTS = [
    "The film was joyful and featured unforgettable performances.",
    "The movie was awful, tedious, and much too long.",
    "The movie was excellent, with strong acting and an engaging plot.",
]

CHOICES = [
    "a positive movie review",
    "a negative movie review",
    "a neutral movie review",
]


def main():
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    scorer = BERTScorer(
        model_type="distilbert-base-uncased",
        device=DEVICE,
    )

    classification = eval_temp_classification(
        model,
        tokenizer,
        MODIFIED_TEXTS,
        CHOICES,
    )
    bertscore = eval_temp_bertscore(
        scorer,
        SOURCE_TEXTS,
        MODIFIED_TEXTS,
    )
    edit_distance = eval_temp_edit_distance(
        SOURCE_TEXTS,
        MODIFIED_TEXTS,
    )
    perplexity = eval_temp_perplexity(
        model,
        tokenizer,
        MODIFIED_TEXTS,
    )

    print("classification:", classification)
    print("BERTScore F1:", bertscore)
    print("normalized edit distance:", edit_distance)
    print("perplexity:", perplexity)


if __name__ == "__main__":
    main()
