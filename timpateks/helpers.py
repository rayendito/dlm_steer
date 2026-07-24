import re

import torch
import torch.nn.functional as F

from .llada.generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
)


LLADA_MASK_TOKEN = "<|mdm_mask|>"


def _as_text_list(text):
    texts = [text] if isinstance(text, str) else text
    if not isinstance(texts, list) or not texts:
        raise ValueError("text must be a string or a non-empty list of strings.")
    if not all(isinstance(item, str) for item in texts):
        raise TypeError("Each text must be a string.")
    return texts


def _as_prompt_list(prompt, count, name="prompt"):
    prompts = [prompt] * count if isinstance(prompt, str) else prompt
    if not isinstance(prompts, list) or len(prompts) != count:
        raise ValueError(f"{name} must be a string or contain one string per text.")
    if not all(isinstance(item, str) for item in prompts):
        raise TypeError(f"Every item in {name} must be a string.")
    return prompts


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_mask_token_id(tokenizer):
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        return mask_token_id

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if LLADA_MASK_TOKEN in vocab:
        return vocab[LLADA_MASK_TOKEN]

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        candidate = tokenizer.convert_tokens_to_ids(LLADA_MASK_TOKEN)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if candidate is not None and candidate != unk_token_id:
            return candidate

    raise ValueError(
        f"The diffusion tokenizer does not define {LLADA_MASK_TOKEN!r}."
    )


def _scores_to_masking_probs(
    scores,
    attention_mask,
    temperature,
    margin,
    mapping="tanh",
):
    """Map negative token scores to masking probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if margin < 0:
        raise ValueError("margin must be greater than or equal to zero.")
    if mapping not in {"tanh", "sigmoid"}:
        raise ValueError("mapping must be 'tanh' or 'sigmoid'.")

    negative_evidence = -scores - margin
    if mapping == "tanh":
        masking_probs = torch.tanh(
            negative_evidence.clamp_min(0) / temperature
        )
    else:
        masking_probs = torch.sigmoid(negative_evidence / temperature)
    if attention_mask is not None:
        masking_probs = masking_probs.masked_fill(attention_mask == 0, 0.0)
    return masking_probs


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


def sample_mask(
    masking_probs,
    tokenizer,
    text,
    attention_mask=None,
    generator=None,
):
    """Sample one masking decision per word and broadcast it to its tokens."""
    if not isinstance(masking_probs, torch.Tensor) or masking_probs.ndim != 2:
        raise ValueError("masking_probs must have shape [batch, tokens].")
    if torch.any((masking_probs < 0) | (masking_probs > 1)):
        raise ValueError("masking_probs values must be between zero and one.")

    texts = _as_text_list(text)
    if len(texts) != masking_probs.shape[0]:
        raise ValueError(
            "The number of texts must match the masking probability batch size."
        )

    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            "Word-level mask sampling requires a fast tokenizer with a pad "
            "token and support for return_offsets_mapping=True."
        ) from exc

    target_offsets = encoded["offset_mapping"].to(masking_probs.device)
    if target_offsets.shape[:2] != masking_probs.shape:
        raise RuntimeError(
            "Retokenized text shape does not match the aligned masking probabilities."
        )
    if attention_mask is None:
        attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(masking_probs, dtype=torch.long)
    attention_mask = attention_mask.to(masking_probs.device).bool()

    masked_positions = torch.zeros_like(masking_probs, dtype=torch.bool)
    for row, item in enumerate(texts):
        row_offsets = target_offsets[row]
        for match in re.finditer(r"\s*\w+(?:['’]\w+)*|[^\w\s]+", item):
            start, end = match.span()
            overlap = (
                torch.minimum(row_offsets[:, 1], row_offsets.new_tensor(end))
                - torch.maximum(row_offsets[:, 0], row_offsets.new_tensor(start))
            ) > 0
            token_group = overlap & attention_mask[row]
            if not token_group.any():
                continue

            # Alignment broadcasts one word probability to all of its tokens.
            # Mean is defensive for the rare tokenizer token crossing a boundary.
            word_probability = masking_probs[row, token_group].mean()
            sampled = torch.rand(
                (),
                device=masking_probs.device,
                generator=generator,
            ) < word_probability
            masked_positions[row, token_group] = sampled

    return masked_positions & attention_mask


def _random_token_detection(
    tokenizer,
    texts,
    probability,
    device,
    generator=None,
):
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("random_mask_probability must be between zero and one.")

    tokenized_text = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(tokenized_text, "to"):
        tokenized_text = tokenized_text.to(device)
    else:
        tokenized_text = {
            key: value.to(device) for key, value in tokenized_text.items()
        }
    attention_mask = tokenized_text.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tokenized_text["input_ids"])
        tokenized_text["attention_mask"] = attention_mask

    masking_probs = torch.full(
        tokenized_text["input_ids"].shape,
        float(probability),
        device=device,
        dtype=torch.float32,
    ).masked_fill(attention_mask == 0, 0.0)
    masked_positions = sample_mask(
        masking_probs,
        tokenizer,
        texts,
        attention_mask=attention_mask,
        generator=generator,
    )
    return tokenized_text, masking_probs, masked_positions


def _probabilistic_token_scores(
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer,
    texts,
    base_assistant_prompt,
    use_chat_template,
):
    """Return LLaDA-tokenized text and aligned AR word log-probability deltas."""
    if identifier_model is None or identifier_tokenizer is None:
        raise ValueError(
            "identifier_model and identifier_tokenizer are required for model detection."
        )

    steer_prompts = _as_prompt_list(steer, len(texts), name="steer")
    base_prompts = _as_prompt_list(
        base_assistant_prompt,
        len(texts),
        name="base_assistant_prompt",
    )
    base_prompt_scores = score_tokens_with_ar(
        identifier_model,
        identifier_tokenizer,
        base_prompts,
        texts,
        use_chat_template=use_chat_template,
    )
    steer_prompt_scores = score_tokens_with_ar(
        identifier_model,
        identifier_tokenizer,
        steer_prompts,
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
    return tokenized_text, aligned_word_log_deltas


def _probabilistic_token_detection_from_scores(
    tokenizer,
    texts,
    tokenized_text,
    aligned_word_log_deltas,
    temperature,
    margin,
    generator=None,
):
    """Map cached aligned scores to probabilities and sample whole-word masks."""
    attention_mask = tokenized_text.get("attention_mask")
    if aligned_word_log_deltas.shape != tokenized_text["input_ids"].shape:
        raise ValueError(
            "aligned_word_log_deltas must match tokenized_text input_ids."
        )
    masking_probs = _scores_to_masking_probs(
        aligned_word_log_deltas,
        attention_mask,
        temperature,
        margin,
        mapping="tanh",
    )
    masked_positions = sample_mask(
        masking_probs,
        tokenizer,
        texts,
        attention_mask=attention_mask,
        generator=generator,
    )
    return tokenized_text, masking_probs, masked_positions


def _probabilistic_token_detection(
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer,
    texts,
    base_assistant_prompt,
    temperature,
    margin,
    use_chat_template,
    generator=None,
):
    tokenized_text, aligned_word_log_deltas = _probabilistic_token_scores(
        tokenizer=tokenizer,
        identifier_model=identifier_model,
        identifier_tokenizer=identifier_tokenizer,
        steer=steer,
        texts=texts,
        base_assistant_prompt=base_assistant_prompt,
        use_chat_template=use_chat_template,
    )
    return _probabilistic_token_detection_from_scores(
        tokenizer=tokenizer,
        texts=texts,
        tokenized_text=tokenized_text,
        aligned_word_log_deltas=aligned_word_log_deltas,
        temperature=temperature,
        margin=margin,
        generator=generator,
    )


def _prepare_steer_vectors(model, steer_vectors, steer_mode):
    if not isinstance(steer_vectors, dict) or not steer_vectors:
        raise ValueError(
            "steer_vectors must be a non-empty {source_layer: direction} mapping."
        )
    if steer_mode not in {"add", "project_out"}:
        raise ValueError("steer_mode must be 'add' or 'project_out'.")
    if steer_mode == "project_out" and len(steer_vectors) != 1:
        raise ValueError(
            "project_out steering requires one selected {source_layer: direction}."
        )

    num_layers = getattr(getattr(model, "config", None), "n_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("The diffusion model config must define a positive n_layers.")
    device = _model_device(model)

    prepared_vectors = {}
    for source_layer, vector in steer_vectors.items():
        if not isinstance(source_layer, int):
            raise TypeError("Each steering-vector source layer must be an integer.")
        max_layer = num_layers if steer_mode == "add" else num_layers - 1
        if not 0 <= source_layer <= max_layer:
            raise ValueError(
                f"source_layer must be between 0 and {max_layer}, inclusive."
            )
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError("Each steering direction must be one-dimensional.")
        if not torch.isfinite(vector).all() or vector.float().norm() == 0:
            raise ValueError("Each steering direction must be finite and non-zero.")
        prepared_vectors[source_layer] = vector.to(
            device=device,
            dtype=torch.float32,
        )

    if steer_mode == "add":
        return prepared_vectors, prepared_vectors

    source_layer, direction = next(iter(prepared_vectors.items()))
    direction = F.normalize(direction, dim=0)
    return (
        {source_layer: direction},
        {layer: direction for layer in range(num_layers)},
    )


@torch.no_grad()
def _steering_token_detection(
    model,
    tokenizer,
    prepared_vectors,
    system_prompts,
    texts,
    use_chat_template,
    temperature,
    margin,
    generator=None,
):
    """Detect response tokens using cosine similarity to steering vectors."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")
    if margin < 0:
        raise ValueError("margin must be greater than or equal to zero.")

    device = _model_device(model)
    tokenized_text = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    response_attention_mask = tokenized_text.get("attention_mask")
    if response_attention_mask is None:
        response_attention_mask = torch.ones_like(tokenized_text["input_ids"])
        tokenized_text["attention_mask"] = response_attention_mask

    sequences = []
    response_masks = []
    for row, prompt in enumerate(system_prompts):
        response_ids = tokenized_text["input_ids"][row][
            response_attention_mask[row].bool()
        ]
        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "use_chat_template=True requires a tokenizer with a chat template."
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

        sequence = torch.cat((prompt_ids, response_ids))
        response_mask = torch.zeros(sequence.numel(), dtype=torch.bool)
        response_mask[prompt_ids.numel():] = True
        sequences.append(sequence)
        response_masks.append(response_mask)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    max_length = max(sequence.numel() for sequence in sequences)
    if pad_token_id is None and any(seq.numel() < max_length for seq in sequences):
        raise ValueError("The diffusion tokenizer must define pad_token_id for batching.")
    pad_token_id = 0 if pad_token_id is None else pad_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")

    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    response_mask_batch = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (sequence, response_mask) in enumerate(zip(sequences, response_masks)):
        start = max_length - sequence.numel() if padding_side == "left" else 0
        end = start + sequence.numel()
        input_ids[row, start:end] = sequence.to(device)
        attention_mask[row, start:end] = 1
        response_mask_batch[row, start:end] = response_mask.to(device)

    if hasattr(tokenized_text, "to"):
        tokenized_text = tokenized_text.to(device)
    else:
        tokenized_text = {
            key: value.to(device) for key, value in tokenized_text.items()
        }
    response_attention_mask = tokenized_text["attention_mask"]

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    sequence_cosine_scores = score_tokens_with_cosine(
        output,
        steer_vectors=prepared_vectors,
    )
    cosine_scores = torch.zeros(
        tokenized_text["input_ids"].shape,
        device=device,
        dtype=sequence_cosine_scores.dtype,
    )
    for row in range(len(texts)):
        cosine_scores[row, response_attention_mask[row].bool()] = (
            sequence_cosine_scores[row, response_mask_batch[row]]
        )

    masking_probs = _scores_to_masking_probs(
        cosine_scores,
        response_attention_mask,
        temperature,
        margin=margin,
        mapping="sigmoid",
    )
    masked_positions = sample_mask(
        masking_probs,
        tokenizer,
        texts,
        attention_mask=response_attention_mask,
        generator=generator,
    )
    return tokenized_text, masking_probs, masked_positions


@torch.no_grad()
def regenerate_masked_text(
    model,
    tokenizer,
    steer,
    text,
    masked_positions,
    response_attention_mask=None,
    use_chat_template=True,
    refill_steps=32,
    sampling_temperature=0.0,
    refill_strategy="low_confidence",
    steer_vectors=None,
    steer_mode="add",
    alpha=1.0,
):
    """Refill sampled response masks with optional activation steering."""
    if refill_steps <= 0:
        raise ValueError("refill_steps must be greater than zero.")
    if sampling_temperature < 0:
        raise ValueError("sampling_temperature must be greater than or equal to zero.")
    if refill_strategy not in {"low_confidence", "random"}:
        raise ValueError("refill_strategy must be 'low_confidence' or 'random'.")
    if steer_mode not in {"add", "project_out"}:
        raise ValueError("steer_mode must be 'add' or 'project_out'.")
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be a finite number.")
    if alpha < 0:
        raise ValueError("alpha must be greater than or equal to zero.")

    texts = _as_text_list(text)
    if not isinstance(steer, list) or len(steer) != len(texts):
        raise ValueError("steer must contain one prompt string per text.")
    if masked_positions.shape[0] != len(texts):
        raise ValueError("masked_positions batch size must match text.")

    mask_token_id = _get_mask_token_id(tokenizer)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = masked_positions.device

    sequences = []
    response_masks = []
    refill_masks = []
    for row, (prompt, item) in enumerate(zip(steer, texts)):
        response_ids = tokenizer(
            item,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        valid_mask = masked_positions[row]
        if response_attention_mask is not None:
            valid_mask = valid_mask[response_attention_mask[row].bool()]
        if valid_mask.numel() != response_ids.numel():
            raise RuntimeError(
                "Sampled mask count does not match the unpadded response token count."
            )

        if use_chat_template:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError(
                    "use_chat_template=True requires a tokenizer with a chat template."
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

        sequence = torch.cat((prompt_ids, response_ids))
        response_mask = torch.zeros(sequence.numel(), dtype=torch.bool)
        response_mask[prompt_ids.numel():] = True
        refill_mask = torch.zeros_like(response_mask)
        refill_mask[prompt_ids.numel():] = valid_mask.cpu()
        sequence[refill_mask] = mask_token_id
        sequences.append(sequence)
        response_masks.append(response_mask)
        refill_masks.append(refill_mask)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    max_length = max(sequence.numel() for sequence in sequences)
    if pad_token_id is None and any(seq.numel() < max_length for seq in sequences):
        raise ValueError("The diffusion tokenizer must define pad_token_id for batching.")
    pad_token_id = 0 if pad_token_id is None else pad_token_id
    padding_side = getattr(tokenizer, "padding_side", "right")

    input_ids = torch.full(
        (len(sequences), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    response_mask_batch = torch.zeros_like(input_ids, dtype=torch.bool)
    refill_mask_batch = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (sequence, response_mask, refill_mask) in enumerate(
        zip(sequences, response_masks, refill_masks)
    ):
        start = max_length - sequence.numel() if padding_side == "left" else 0
        end = start + sequence.numel()
        input_ids[row, start:end] = sequence.to(device)
        attention_mask[row, start:end] = 1
        response_mask_batch[row, start:end] = response_mask.to(device)
        refill_mask_batch[row, start:end] = refill_mask.to(device)

    x = input_ids.clone()
    num_transfer_tokens = get_num_transfer_tokens(refill_mask_batch, refill_steps)
    for refill_step in range(refill_steps):
        still_masked = (x == mask_token_id) & refill_mask_batch
        if not still_masked.any():
            break

        model_kwargs = {}
        if steer_vectors is not None:
            model_kwargs = {
                "steers": steer_vectors,
                "steer_mask": attention_mask,
                "steer_mode": steer_mode,
                "steer_alpha": float(alpha),
            }
        logits = model(
            input_ids=x,
            attention_mask=attention_mask,
            **model_kwargs,
        ).logits
        noisy_logits = add_gumbel_noise(logits, temperature=sampling_temperature)
        candidates = torch.argmax(noisy_logits, dim=-1)
        if refill_strategy == "low_confidence":
            candidate_probs = torch.softmax(logits.float(), dim=-1).gather(
                dim=-1,
                index=candidates.unsqueeze(-1),
            ).squeeze(-1)
        else:
            candidate_probs = torch.rand(x.shape, device=x.device)

        confidence = candidate_probs.masked_fill(~still_masked, -torch.inf)
        transfer = torch.zeros_like(still_masked)
        for row in range(x.shape[0]):
            count = min(
                int(num_transfer_tokens[row, refill_step].item()),
                int(still_masked[row].sum().item()),
            )
            if count > 0:
                indices = torch.topk(confidence[row], k=count).indices
                transfer[row, indices] = True
        x[transfer] = candidates[transfer]

    regenerated_texts = [
        tokenizer.decode(
            x[row, response_mask_batch[row]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row in range(x.shape[0])
    ]
    return regenerated_texts

@torch.no_grad()
def score_tokens_with_cosine(out, steer_vectors):
    """Return mean per-token cosine similarity across steering layers."""
    if not isinstance(steer_vectors, dict) or not steer_vectors:
        raise ValueError("steer_vectors must be a non-empty {layer: vector} mapping.")

    sims = []
    for steer_idx, svector in steer_vectors.items():
        h = out.hidden_states[steer_idx]
        sim = F.cosine_similarity(
            h,
            svector.to(device=h.device, dtype=h.dtype).view(1, 1, -1),
            dim=-1,
        )
        sims.append(sim)
    return torch.stack(sims, dim=0).mean(dim=0)
