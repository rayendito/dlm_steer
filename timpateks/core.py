import torch
import torch.nn.functional as F

@torch.no_grad()
def score_tokens_wrt_steer(
    model,
    tokenizer,
    steer,
    text,
    identifier_mode=None,
):
    """Score each token in ``text`` against steering entities.

    For cosine scoring, ``steer`` is a dictionary mapping hidden-state layer
    indices to one-dimensional steering tensors.

    For conditional-probability
    scoring, it is a list of string prompts paired one-to-one with ``text``.
    ``identifier_mode="AR"`` uses
    shifted next-token probabilities, while ``identifier_mode="DLM"`` masks
    each text token and computes its pseudo-likelihood. Cosine scores are
    averaged over the supplied layers. Only tokens belonging to ``text`` are
    scored.

    Returns ``(scores, text_token_indices)``. Both tensors have shape
    ``[batch_size, sequence_length]``. Padding positions have a score of zero
    and an index of ``-1``. For conditional-probability scoring, indices refer
    to token positions in the combined prompt-and-text input; for cosine
    scoring, they refer to positions in the tokenized text input.

    ``text`` may be either a string or a batch of strings.
    """
    method = "cosine" if isinstance(steer, dict) else "cond_prob"

    if method == "cond_prob":
        if not isinstance(steer, list) or not steer:
            raise ValueError("steer must be a non-empty list of prompt strings.")
        if not all(isinstance(prompt, str) for prompt in steer):
            raise TypeError("Each conditional-probability steer must be a string.")

        texts = [text] if isinstance(text, str) else text
        if not isinstance(texts, list) or not texts:
            raise ValueError("text must be a string or a non-empty list of strings.")
        if not all(isinstance(item, str) for item in texts):
            raise TypeError("Each text must be a string.")
        if len(steer) != len(texts):
            raise ValueError(
                "Conditional-probability scoring requires one steer per text; "
                f"received {len(steer)} steers and {len(texts)} texts."
            )

        if identifier_mode is None:
            raise ValueError(
                "identifier_mode must be either 'AR' or 'DLM' for "
                "conditional-probability scoring."
            )
        identifier_mode = identifier_mode.upper()
        if identifier_mode not in {"AR", "DLM"}:
            raise ValueError("identifier_mode must be either 'AR' or 'DLM'.")

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        if identifier_mode == "DLM":
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError(
                    "The identifier tokenizer must define mask_token_id for "
                    "DLM scoring."
                )

        scores_by_text = []
        indices_by_text = []
        for prompt, item in zip(steer, texts):
            text_ids = tokenizer(
                item,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if text_ids.numel() == 0:
                scores_by_text.append(torch.empty(0, device=device))
                indices_by_text.append(
                    torch.empty(0, device=device, dtype=torch.long)
                )
                continue

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            combined_ids = torch.cat((prompt_ids, text_ids)).to(device)
            text_length = text_ids.numel()
            text_positions = (
                prompt_ids.numel()
                + torch.arange(text_length, device=device)
            )
            targets = text_ids.to(device)

            if identifier_mode == "AR":
                prediction_positions = text_positions - 1
                if prediction_positions[0] < 0:
                    raise ValueError(
                        "An AR steer prompt must contain at least one token "
                        "to predict the first text token."
                    )
                input_ids = combined_ids.unsqueeze(0)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                ).logits
                text_logits = logits[0, prediction_positions, :]

            else:  # DLM
                input_ids = combined_ids.unsqueeze(0).repeat(text_length, 1)
                rows = torch.arange(text_length, device=device)
                input_ids[rows, text_positions] = mask_token_id
                logits = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                ).logits
                text_logits = logits[rows, text_positions, :]

            token_probs = torch.gather(
                torch.softmax(text_logits.float(), dim=-1),
                dim=-1,
                index=targets.unsqueeze(-1),
            ).squeeze(-1)
            scores_by_text.append(token_probs)
            indices_by_text.append(text_positions)

        max_length = max(scores.numel() for scores in scores_by_text)
        scores = torch.zeros(
            (len(scores_by_text), max_length),
            device=device,
            dtype=torch.float32,
        )
        text_token_indices = torch.full(
            (len(scores_by_text), max_length),
            -1,
            device=device,
            dtype=torch.long,
        )
        for row, (item_scores, item_indices) in enumerate(
            zip(scores_by_text, indices_by_text)
        ):
            scores[row, :item_scores.numel()] = item_scores
            text_token_indices[row, :item_indices.numel()] = item_indices
        return scores, text_token_indices

    elif method == "cosine":
        if not steer:
            raise ValueError("steer must contain at least one layer and vector.")

        for layer_idx, vector in steer.items():
            if not isinstance(layer_idx, int):
                raise TypeError("steer layer indices must be integers.")
            if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
                raise ValueError(
                    "Each steer value must be a one-dimensional tensor; "
                    f"layer {layer_idx} has {type(vector).__name__} with shape "
                    f"{getattr(vector, 'shape', None)}."
                )

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = next(iter(steer.values())).device

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(encoded["input_ids"])

        out = model(**encoded, output_hidden_states=True)
        if out.hidden_states is None:
            raise RuntimeError("The model did not return hidden states.")

        similarities = []
        for layer_idx, vector in steer.items():
            if not -len(out.hidden_states) <= layer_idx < len(out.hidden_states):
                raise IndexError(
                    f"Steer layer index {layer_idx} is out of range for "
                    f"{len(out.hidden_states)} hidden-state layers."
                )

            hidden = out.hidden_states[layer_idx]
            if hidden.shape[-1] != vector.shape[-1]:
                raise ValueError(
                    f"Steering-vector size {vector.shape[-1]} at layer "
                    f"{layer_idx} does not match model hidden size "
                    f"{hidden.shape[-1]}."
                )
            vector = vector.to(device=hidden.device, dtype=hidden.dtype)
            similarities.append(
                F.cosine_similarity(hidden, vector.view(1, 1, -1), dim=-1)
            )

        scores = torch.stack(similarities, dim=0).mean(dim=0)
        scores = scores.masked_fill(attention_mask == 0, 0.0)
        text_token_indices = torch.arange(
            scores.shape[1], device=scores.device
        ).expand_as(attention_mask).masked_fill(attention_mask == 0, -1)
        return scores, text_token_indices
