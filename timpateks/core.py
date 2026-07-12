import re

import torch


def _as_text_list(text):
    texts = [text] if isinstance(text, str) else text
    if not isinstance(texts, list) or not texts:
        raise ValueError("text must be a string or a non-empty list of strings.")
    if not all(isinstance(item, str) for item in texts):
        raise TypeError("Each text must be a string.")
    return texts


def _text_encoding(tokenizer, text):
    try:
        return tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            "Token alignment requires a fast tokenizer that supports "
            "return_offsets_mapping=True."
        ) from exc


def _sum_token_scores_by_word(token_scores, token_offsets, texts):
    """Sum token scores within words and return padded word-level tensors."""
    scores_by_text = []
    offsets_by_text = []
    scores_cpu = token_scores.detach().float().cpu()
    offsets_cpu = token_offsets.cpu()

    for row, text in enumerate(texts):
        valid = offsets_cpu[row, :, 1] > offsets_cpu[row, :, 0]
        source_scores = scores_cpu[row, valid]
        source_offsets = offsets_cpu[row, valid]
        word_scores = []
        word_offsets = []

        # Include leading whitespace in the following word, matching tokens
        # such as " th" produced by byte-level BPE tokenizers.
        for match in re.finditer(r"\s*\w+(?:['’]\w+)*|[^\w\s]+", text):
            start, end = match.span()
            overlaps_word = (
                torch.minimum(source_offsets[:, 1], source_offsets.new_tensor(end))
                - torch.maximum(source_offsets[:, 0], source_offsets.new_tensor(start))
            ) > 0
            if overlaps_word.any():
                word_scores.append(source_scores[overlaps_word].sum())
                word_offsets.append((start, end))

        scores_by_text.append(
            torch.stack(word_scores) if word_scores else torch.empty(0)
        )
        offsets_by_text.append(
            torch.tensor(word_offsets, dtype=torch.long).reshape(-1, 2)
        )

    max_words = max(scores.numel() for scores in scores_by_text)
    grouped_scores = torch.zeros(
        (len(texts), max_words),
        device=token_scores.device,
        dtype=token_scores.dtype,
    )
    grouped_offsets = torch.full(
        (len(texts), max_words, 2), -1, dtype=torch.long
    )
    for row, (scores, offsets) in enumerate(zip(scores_by_text, offsets_by_text)):
        grouped_scores[row, :scores.numel()] = scores.to(
            device=grouped_scores.device, dtype=grouped_scores.dtype
        )
        grouped_offsets[row, :offsets.shape[0]] = offsets
    return grouped_scores, grouped_offsets


@torch.no_grad()
def score_tokens_with_ar(
    model,
    tokenizer,
    steer,
    text,
    use_chat_template=True,
):
    """Score response tokens with an autoregressive identifier model.

    ``steer`` is a list of system prompts paired one-to-one with ``text``.
    When ``use_chat_template`` is true, each prompt is rendered as a system
    message followed by the tokenizer's assistant generation prompt.

    Returns ``(scores, offsets)``. ``scores`` has shape ``[batch, tokens]``
    and contains next-token probabilities. ``offsets`` has shape
    ``[batch, tokens, 2]`` and contains character spans relative to the raw
    response text. Padded scores are zero and padded offsets are ``(-1, -1)``.
    """
    if not isinstance(steer, list) or not steer:
        raise ValueError("steer must be a non-empty list of prompt strings.")
    if not all(isinstance(prompt, str) for prompt in steer):
        raise TypeError("Each steer prompt must be a string.")

    texts = _as_text_list(text)
    if len(steer) != len(texts):
        raise ValueError(
            "AR scoring requires one steer per text; "
            f"received {len(steer)} steers and {len(texts)} texts."
        )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    scores_by_text = []
    offsets_by_text = []
    for prompt, item in zip(steer, texts):
        encoded_text = _text_encoding(tokenizer, item)
        text_ids = encoded_text["input_ids"][0]
        text_offsets = encoded_text["offset_mapping"][0].to(torch.long)
        if text_ids.numel() == 0:
            scores_by_text.append(torch.empty(0, device=device))
            offsets_by_text.append(text_offsets)
            continue

        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "use_chat_template=True requires a tokenizer with a "
                    "chat template."
                )
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(prompt_ids, dict):
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = prompt_ids[0]
        else:
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]

        if prompt_ids.numel() == 0:
            raise ValueError(
                "An AR steer prompt must contain at least one token so the "
                "first response token has a preceding prediction position."
            )

        combined_ids = torch.cat((prompt_ids, text_ids)).to(device)
        text_positions = prompt_ids.numel() + torch.arange(
            text_ids.numel(), device=device
        )
        input_ids = combined_ids.unsqueeze(0)
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        ).logits
        text_logits = logits[0, text_positions - 1, :]
        targets = text_ids.to(device)
        token_probs = torch.gather(
            torch.softmax(text_logits.float(), dim=-1),
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)
        scores_by_text.append(token_probs)
        offsets_by_text.append(text_offsets)

    max_length = max(scores.numel() for scores in scores_by_text)
    scores = torch.zeros(
        (len(scores_by_text), max_length), device=device, dtype=torch.float32
    )
    offsets = torch.full(
        (len(scores_by_text), max_length, 2), -1, dtype=torch.long
    )
    for row, (item_scores, item_offsets) in enumerate(
        zip(scores_by_text, offsets_by_text)
    ):
        length = item_scores.numel()
        scores[row, :length] = item_scores
        offsets[row, :length] = item_offsets
    return scores, offsets


def tokenize_and_align_ar_scores(
    ar_token_scores,
    tokenizer,
    text,
    use_chat_template=True,
):
    """Tokenize text and align AR-derived scores to the resulting tokens.

    Every target token receives the character-length-weighted mean of all AR
    token scores that overlap it. The chat-template argument is accepted to
    mirror the surrounding API; alignment itself operates only on raw response
    text, so prompt and assistant-header tokens never enter the mapping.

    Returns ``(tokenized_text, aligned_scores)``. ``tokenized_text`` is the
    tokenizer's model-ready batch encoding, and ``aligned_scores`` follows its
    ``[batch, target_tokens]`` shape and padding positions.
    """
    del use_chat_template
    if (
        not isinstance(ar_token_scores, (tuple, list))
        or len(ar_token_scores) != 2
    ):
        raise TypeError("ar_token_scores must be a (scores, offsets) pair.")
    ar_scores, ar_offsets = ar_token_scores
    if not isinstance(ar_scores, torch.Tensor) or ar_scores.ndim != 2:
        raise ValueError("AR scores must have shape [batch, tokens].")
    if (
        not isinstance(ar_offsets, torch.Tensor)
        or ar_offsets.ndim != 3
        or ar_offsets.shape[-1] != 2
        or ar_offsets.shape[:2] != ar_scores.shape
    ):
        raise ValueError("AR offsets must have shape [batch, tokens, 2].")

    texts = _as_text_list(text)
    if len(texts) != ar_scores.shape[0]:
        raise ValueError(
            "The number of texts must match the AR score batch size; "
            f"received {len(texts)} texts and {ar_scores.shape[0]} score rows."
        )

    try:
        tokenized_text = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            "Token alignment requires a fast target tokenizer with a pad token "
            "and support for return_offsets_mapping=True."
        ) from exc

    target_offsets = tokenized_text.pop("offset_mapping").to(torch.long)
    target_shape = tokenized_text["input_ids"].shape
    aligned_scores = torch.zeros(
        target_shape, device=ar_scores.device, dtype=ar_scores.dtype
    )

    # Offset tensors are tiny and tokenizer output is CPU-resident. Keeping the
    # overlap calculation on CPU avoids unnecessary device synchronization per
    # token; only the completed row is copied to the score device.
    ar_offsets_cpu = ar_offsets.cpu()
    ar_scores_cpu = ar_scores.detach().float().cpu()
    for row, row_target_offsets in enumerate(target_offsets):
        valid_ar = ar_offsets_cpu[row, :, 1] > ar_offsets_cpu[row, :, 0]
        source_spans = ar_offsets_cpu[row, valid_ar]
        source_scores = ar_scores_cpu[row, valid_ar]
        row_scores = torch.zeros(row_target_offsets.shape[0], dtype=torch.float32)

        for target_index, (target_start, target_end) in enumerate(
            row_target_offsets
        ):
            if target_end <= target_start or source_spans.numel() == 0:
                continue
            overlap = (
                torch.minimum(source_spans[:, 1], target_end)
                - torch.maximum(source_spans[:, 0], target_start)
            ).clamp_min(0)
            overlap_total = overlap.sum()
            if overlap_total > 0:
                row_scores[target_index] = (
                    source_scores * overlap.to(source_scores.dtype)
                ).sum() / overlap_total

        aligned_scores[row, : row_scores.numel()] = row_scores.to(
            device=aligned_scores.device, dtype=aligned_scores.dtype
        )
    if hasattr(tokenized_text, "to"):
        tokenized_text = tokenized_text.to(ar_scores.device)
    else:
        tokenized_text = {
            key: value.to(ar_scores.device)
            for key, value in tokenized_text.items()
        }
    return tokenized_text, aligned_scores


def timpa(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer,
    text,
    use_chat_template=True,
    base_assistant_prompt="You are a helpful assistant",
    temperature=1.0,
    margin=0.001,
):
    """Compute and align word-level AR log-probability changes.

    AR-token log-probability changes are summed within each word and broadcast
    to every diffusion token overlapping that word. They are then mapped to
    masking probabilities after a margin-based dead zone. Words whose log
    delta is at least ``-margin`` receive zero masking probability.
    """
    del model  # Reserved for the steering logic that follows identification.
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if margin < 0:
        raise ValueError("margin must be greater than or equal to zero.")

    texts = _as_text_list(text)
    base_prompt_scores = score_tokens_with_ar(
        identifier_model,
        identifier_tokenizer,
        [base_assistant_prompt] * len(texts),
        texts,
        use_chat_template=use_chat_template,
    )
    steer_prompt_scores = score_tokens_with_ar(
        identifier_model,
        identifier_tokenizer,
        steer,
        texts,
        use_chat_template=use_chat_template,
    )

    base_probs, base_offsets = base_prompt_scores
    steer_probs, steer_offsets = steer_prompt_scores
    if not torch.equal(base_offsets, steer_offsets):
        raise RuntimeError(
            "AR token offsets changed between the base and steer prompt scores."
        )
    epsilon = torch.finfo(steer_probs.dtype).tiny
    token_log_deltas = (
        steer_probs.clamp_min(epsilon).log()
        - base_probs.clamp_min(epsilon).log()
    )
    word_log_deltas, word_offsets = _sum_token_scores_by_word(
        token_log_deltas,
        steer_offsets,
        texts,
    )

    tokenized_text, aligned_word_log_deltas = tokenize_and_align_ar_scores(
        (word_log_deltas, word_offsets),
        tokenizer,
        texts,
        use_chat_template=use_chat_template,
    )

    negative_evidence = (-aligned_word_log_deltas - margin).clamp_min(0)

    masking_probs = torch.tanh(negative_evidence / (2 * temperature))

    attention_mask = tokenized_text.get("attention_mask")
    if attention_mask is not None:
        masking_probs = masking_probs.masked_fill(attention_mask == 0, 0.0)

    return tokenized_text, masking_probs
