def see_tokens_from_ids(tokenizer, input_ids):
    """Print token indices, IDs, raw tokens, and decoded text for one or more sequences."""
    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().tolist()
    elif hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    if not isinstance(input_ids, (list, tuple)):
        raise TypeError("input_ids must be a token-ID sequence or a batch of sequences.")

    is_single_sequence = not input_ids or isinstance(input_ids[0], int)
    sequences = [input_ids] if is_single_sequence else input_ids

    for batch_idx, sequence in enumerate(sequences):
        if not isinstance(sequence, (list, tuple)) or not all(
            isinstance(token_id, int) for token_id in sequence
        ):
            raise TypeError("input_ids must contain integers and have at most two dimensions.")

        if len(sequences) > 1:
            print(f"Sequence {batch_idx}")
        print(f"{'index':>5} {'token_id':>8}  {'token':<28} decoded")

        tokens = tokenizer.convert_ids_to_tokens(list(sequence))
        for index, (token_id, token) in enumerate(zip(sequence, tokens)):
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            print(f"{index:>5} {token_id:>8}  {token!r:<28} {decoded!r}")

        if batch_idx < len(sequences) - 1:
            print()
