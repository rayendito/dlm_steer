import torch
import torch.nn.functional as F


NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."


def _as_text_list(text, name):
    if not isinstance(text, list) or not text:
        raise ValueError(f"{name} must be a non-empty list of strings.")
    if not all(isinstance(item, str) for item in text):
        raise TypeError(f"Every item in {name} must be a string.")
    return text


def _validate_text_pairs(text_before, text_after):
    before = _as_text_list(text_before, "text_before")
    after = _as_text_list(text_after, "text_after")
    if len(before) != len(after):
        raise ValueError("text_before and text_after must have the same length.")
    return before, after


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _chat_ids(tokenizer, messages):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer must define a chat template.")
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("The chat template must produce a non-empty token sequence.")
    return input_ids


def _pad_sequences(sequences, tokenizer, device):
    max_length = max(sequence.numel() for sequence in sequences)
    pad_token_id = _pad_token_id(tokenizer)

    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    starts = []
    for row, sequence in enumerate(sequences):
        # Causal generation should be left-padded so every continuation starts
        # immediately after the final non-padding prompt token.
        start = max_length - sequence.numel()
        end = start + sequence.numel()
        input_ids[row, start:end] = sequence.to(device)
        attention_mask[row, start:end] = 1
        starts.append(start)
    return input_ids, attention_mask, starts


def _pad_token_id(tokenizer):
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id or eos_token_id.")
    return pad_token_id


def _classifier_prompt(text, choices):
    options = "\n".join(f"- {choice}" for choice in choices)
    return [
        {
            "role": "system",
            "content": (
                "You are a strict text classifier. Choose exactly one category "
                "and respond with its category text exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Categories:\n{options}\n\n"
                f"Text:\n{text}\n\n"
                "Category:"
            ),
        },
    ]


@torch.no_grad()
def eval_temp_classification(model, tokenizer, text_after, choices):
    """Return class probabilities from an instruct causal LM.

    ``choices`` is a shared ordered list of category labels. If every label is
    one token, one normal batched forward scores all classes from the next-token
    logits. Otherwise, one expanded ``batch_size * num_classes`` forward scores
    the complete label sequences with teacher forcing.

    Returns a ``[batch_size][num_classes]`` list whose rows sum to one.
    """
    texts = _as_text_list(text_after, "text_after")
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("choices must contain at least two category strings.")
    if not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise TypeError("Every choice must be a non-empty string.")
    if len(set(choices)) != len(choices):
        raise ValueError("choices must not contain duplicates.")

    prompt_ids = [
        _chat_ids(tokenizer, _classifier_prompt(text, choices))
        for text in texts
    ]
    label_ids = [
        tokenizer.encode(choice, add_special_tokens=False)
        for choice in choices
    ]
    if any(not ids for ids in label_ids):
        raise ValueError("Every class label must produce tokens.")
    label_ids = [
        torch.tensor(ids, dtype=torch.long)
        for ids in label_ids
    ]

    device = _model_device(model)
    was_training = model.training
    model.eval()
    try:
        if all(ids.numel() == 1 for ids in label_ids):
            input_ids, attention_mask, _ = _pad_sequences(
                prompt_ids,
                tokenizer,
                device,
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            class_token_ids = torch.tensor(
                [int(ids.item()) for ids in label_ids],
                dtype=torch.long,
                device=device,
            )
            class_log_scores = logits[:, -1].float().index_select(
                dim=-1,
                index=class_token_ids,
            )
        else:
            expanded_sequences = []
            label_lengths = []
            for prompt in prompt_ids:
                for label in label_ids:
                    expanded_sequences.append(torch.cat((prompt, label)))
                    label_lengths.append(label.numel())

            input_ids, attention_mask, starts = _pad_sequences(
                expanded_sequences,
                tokenizer,
                device,
            )
            label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, (start, label_length) in enumerate(
                zip(starts, label_lengths)
            ):
                label_start = start + expanded_sequences[row].numel() - label_length
                label_mask[row, label_start:label_start + label_length] = True

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            shift_log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
            shift_labels = input_ids[:, 1:]
            token_log_probs = shift_log_probs.gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)
            sequence_log_scores = token_log_probs.masked_fill(
                ~label_mask[:, 1:],
                0.0,
            ).sum(dim=-1)
            class_log_scores = sequence_log_scores.view(
                len(texts),
                len(choices),
            )
    finally:
        if was_training:
            model.train()

    class_probs = torch.softmax(class_log_scores, dim=-1)
    return class_probs.detach().cpu().tolist()


def eval_temp_bertscore(scorer, text_before, text_after):
    """Return per-example BERTScore F1 for rewritten text against source text."""
    before, after = _validate_text_pairs(text_before, text_after)
    if scorer is None or not callable(getattr(scorer, "score", None)):
        raise TypeError("scorer must provide a callable score(cands, refs) method.")

    scores = scorer.score(after, before)
    if not isinstance(scores, (tuple, list)) or len(scores) != 3:
        raise TypeError("BERTScore scorer.score must return (precision, recall, F1).")
    f1 = scores[2]
    if isinstance(f1, torch.Tensor):
        values = f1.detach().float().cpu().tolist()
    else:
        values = [float(value) for value in f1]
    if len(values) != len(before):
        raise RuntimeError("BERTScore returned the wrong number of scores.")
    return values


def _levenshtein_distance(first, second):
    if len(first) < len(second):
        first, second = second, first
    if not second:
        return len(first)

    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            insertion = current[second_index - 1] + 1
            deletion = previous[second_index] + 1
            substitution = (
                previous[second_index - 1]
                + (first_character != second_character)
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def eval_temp_edit_distance(text_before, text_after):
    """Return character Levenshtein distance normalized by the longer text."""
    before, after = _validate_text_pairs(text_before, text_after)
    distances = []
    for source, candidate in zip(before, after):
        denominator = max(len(source), len(candidate))
        if denominator == 0:
            distances.append(0.0)
        else:
            distances.append(
                _levenshtein_distance(source, candidate) / denominator
            )
    return distances


@torch.no_grad()
def eval_temp_perplexity(model, tokenizer, text_after):
    """Return per-example assistant-response perplexity from an instruct LM.

    The neutral system prompt and assistant generation prefix provide the chat
    context. Cross-entropy is averaged only over tokens belonging to the
    supplied response, excluding the system and assistant-header tokens.
    """
    texts = _as_text_list(text_after, "text_after")
    prefix_ids = _chat_ids(
        tokenizer,
        [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT}],
    )
    response_ids = [
        tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        for text in texts
    ]
    sequences = [
        torch.cat((prefix_ids, response))
        for response in response_ids
    ]

    device = _model_device(model)
    input_ids, attention_mask, starts = _pad_sequences(
        sequences,
        tokenizer,
        device,
    )
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (start, response) in enumerate(zip(starts, response_ids)):
        response_start = start + prefix_ids.numel()
        response_end = response_start + response.numel()
        response_mask[row, response_start:response_end] = True

    was_training = model.training
    model.eval()
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    finally:
        if was_training:
            model.train()

    shift_logits = logits[:, :-1].float()
    shift_labels = input_ids[:, 1:]
    shift_response_mask = response_mask[:, 1:]
    losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
    )
    token_counts = shift_response_mask.sum(dim=1)
    mean_losses = (
        losses.masked_fill(~shift_response_mask, 0.0).sum(dim=1)
        / token_counts.clamp_min(1)
    )
    perplexities = torch.exp(mean_losses)
    perplexities = perplexities.masked_fill(token_counts == 0, torch.inf)
    return perplexities.detach().cpu().tolist()
