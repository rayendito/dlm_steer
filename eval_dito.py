import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)

def score_labels(batch_texts):
    prompts = [f"Text: {t}\nSentiment:" for t in batch_texts]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits  # [B, T, V]

    # get logits for NEXT token after prompt
    last_token_idx = inputs["attention_mask"].sum(dim=1) - 1
    next_token_logits = logits[torch.arange(len(batch_texts)), last_token_idx]

    probs = F.softmax(next_token_logits, dim=-1)

    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    p_pos = probs[:, pos_id]
    p_neg = probs[:, neg_id]

    preds = ["positive" if p_pos[i] > p_neg[i] else "negative"
             for i in range(len(batch_texts))]

    return preds, p_pos, p_neg


def perplexity(batch_texts):
    # Empty / whitespace-only strings tokenize to length 0 and crash Qwen2 (position_ids view).
    raw = [("" if t is None else str(t)) for t in batch_texts]
    empty_ix = [i for i, t in enumerate(raw) if not t.strip()]
    safe = [t if t.strip() else "." for t in raw]
    enc = tokenizer(safe, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**enc)
        shift_logits = out.logits[:, :-1]
        shift_labels = enc["input_ids"][:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none"
        ).view(shift_labels.shape)

        mask = enc["attention_mask"][:, 1:]
        mask_sum = mask.sum(dim=1).clamp(min=1)
        loss = (loss * mask).sum(dim=1) / mask_sum

        out_ppl = torch.exp(loss)
        for i in empty_ix:
            out_ppl[i] = float("inf")
        return out_ppl

if __name__ == "__main__":
    # quick local sanity check
    texts = [
        "i love this movie! its the best movie ive ever watched",
        "lask as diwoowo d fe fiefhieh feh eu hfuef",
    ]
    preds, p_pos, p_neg = score_labels(texts)
    ppl = perplexity(texts)
    for i, t in enumerate(texts):
        print(
            t,
            preds[i].item() if hasattr(preds[i], "item") else preds[i],
            p_pos[i].item(),
            p_neg[i].item(),
            ppl[i].item(),
        )