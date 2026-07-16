import torch
import torch.nn.functional as F
from timpa_debug import see_tokens_from_ids

@torch.no_grad()
def extract_steer_vectors(
    model,
    tokenizer,
    corpus1,
    corpus2,
    source_layer=23,
    token_position=-4,
    batch_size=8,
):
    """Extract one contrastive direction from a chat-template token.

    Each corpus item is appended as an assistant response to a chat-templated
    empty system prompt. Activations are collected from ``token_position`` in
    the assistant generation prefix at ``source_layer`` and averaged within
    each corpus. The returned direction is ``normalize(mean(corpus1) -
    mean(corpus2))`` in the ``{source_layer: vector}`` format consumed by
    :func:`timpateks.timpa_steer`.

    Non-negative token positions are relative to the start of the unpadded chat
    prefix; negative positions are relative to the end of that prefix, before
    the response is appended. The defaults correspond to the post-instruction
    region used for LLaDA in the paper.
    """
    for name, corpus in (("corpus1", corpus1), ("corpus2", corpus2)):
        if not isinstance(corpus, list) or not corpus:
            raise ValueError(f"{name} must be a non-empty list of strings.")
        if not all(isinstance(text, str) for text in corpus):
            raise TypeError(f"Every item in {name} must be a string.")
    if not isinstance(source_layer, int) or source_layer < 0:
        raise ValueError("source_layer must be a non-negative integer.")
    if not isinstance(token_position, int):
        raise TypeError("token_position must be an integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer must define a chat template.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    def render_prompt(text):
        prefix_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(prefix_ids, dict):
            prefix_ids = prefix_ids["input_ids"]
        prefix_ids = prefix_ids[0]
        if prefix_ids.numel() == 0:
            raise ValueError("A chat-templated extraction prompt cannot be empty.")
        response_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        position = (
            token_position
            if token_position >= 0
            else prefix_ids.numel() + token_position
        )
        if not 0 <= position < prefix_ids.numel():
            raise ValueError(
                f"token_position {token_position} is out of range for a rendered "
                f"prefix containing {prefix_ids.numel()} tokens."
            )
        return torch.cat((prefix_ids, response_ids)), position

    def corpus_mean(corpus):
        activation_sum = None
        sample_count = 0
        for start in range(0, len(corpus), batch_size):
            rendered = [render_prompt(text) for text in corpus[start:start + batch_size]]
            sequences = [item[0] for item in rendered]
            positions = [item[1] for item in rendered]
            max_length = max(sequence.numel() for sequence in sequences)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None and any(
                sequence.numel() < max_length for sequence in sequences
            ):
                raise ValueError("The tokenizer must define pad_token_id for batching.")
            pad_token_id = 0 if pad_token_id is None else pad_token_id
            padding_side = getattr(tokenizer, "padding_side", "right")

            input_ids = torch.full(
                (len(sequences), max_length),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            batch_positions = []
            for row, (sequence, position) in enumerate(zip(sequences, positions)):
                offset = max_length - sequence.numel() if padding_side == "left" else 0
                end = offset + sequence.numel()
                input_ids[row, offset:end] = sequence.to(device)
                attention_mask[row, offset:end] = 1
                batch_positions.append(offset + position)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = output.hidden_states
            if hidden_states is None or source_layer >= len(hidden_states):
                available = 0 if hidden_states is None else len(hidden_states)
                raise ValueError(
                    f"source_layer {source_layer} is unavailable; the model returned "
                    f"{available} hidden-state tensors."
                )
            row_indices = torch.arange(len(sequences), device=device)
            position_indices = torch.tensor(batch_positions, device=device)
            activations = hidden_states[source_layer][
                row_indices,
                position_indices,
            ].float()
            chunk_sum = activations.sum(dim=0).cpu()
            activation_sum = (
                chunk_sum if activation_sum is None else activation_sum + chunk_sum
            )
            sample_count += len(sequences)

        return activation_sum / sample_count

    mean1 = corpus_mean(corpus1)
    mean2 = corpus_mean(corpus2)
    difference = mean1 - mean2
    if not torch.isfinite(difference).all() or difference.norm() == 0:
        raise ValueError("The contrastive corpora produced a non-finite or zero direction.")
    direction = F.normalize(difference, dim=0)
    return {source_layer: direction}
