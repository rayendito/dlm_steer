import torch
import torch.nn.functional as F

@torch.no_grad()
def score_tokens_wrt_steer(model, tokenizer, steer, text):
    """Score each token in ``text`` against steering vectors.

    For cosine scoring, ``steer`` is a dictionary mapping hidden-state layer
    indices to one-dimensional steering tensors. Per-layer cosine similarities
    are averaged and returned without applying a sigmoid or sampling a mask.
    The returned tensor has shape ``[batch_size, sequence_length]``; padding
    positions have a score of zero.

    ``text`` may be either a string or a batch of strings.
    """
    method = "cosine" if isinstance(steer, dict) else "cond_prob"

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    if method == "cond_prob":
        # TODO: implement conditional-probability token scoring.
        pass

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
        return scores.masked_fill(attention_mask == 0, 0.0)
