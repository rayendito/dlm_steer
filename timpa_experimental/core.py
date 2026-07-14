import html
from pathlib import Path

from timpateks import score_tokens_with_ar, timpa_probabilistic


def _highlighted_text(
    text,
    offsets,
    scores,
    token_indices,
):
    valid_scores = scores[:len(offsets)].float().cpu()
    strengths = valid_scores.clamp(0, 1)

    pieces = []
    cursor = 0
    for offset, score, strength, token_index in zip(
        offsets, valid_scores, strengths, token_indices
    ):
        start, end = offset
        if end <= start:
            continue
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        alpha = float(strength)
        pieces.append(
            '<span class="token" '
            f'style="background-color: rgba(255, 140, 0, {alpha:.3f})" '
            f'data-tooltip="probability: {float(score):.6g} · '
            f'input index: {int(token_index)}" '
            f'title="probability: {float(score):.6g} · input index: {int(token_index)}">'
            f'{html.escape(text[start:end])}</span>'
        )
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _comparison_text(text, offsets, first_scores, second_scores):
    first_scores = first_scores[:len(offsets)].float().cpu()
    second_scores = second_scores[:len(offsets)].float().cpu()
    differences = second_scores - first_scores
    scale = max(float(differences.abs().max()), 1e-12)

    pieces = []
    cursor = 0
    for offset, first, second, difference in zip(
        offsets, first_scores, second_scores, differences
    ):
        start, end = offset
        if end <= start:
            continue
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))
        delta = float(difference)
        alpha = abs(delta) / scale
        color = "37, 99, 235" if delta >= 0 else "220, 38, 38"
        tooltip = (
            f"prompt 1: {float(first):.6g} · prompt 2: {float(second):.6g} · "
            f"difference: {delta:+.6g}"
        )
        pieces.append(
            '<span class="token" '
            f'style="background-color: rgba({color}, {alpha:.3f})" '
            f'data-tooltip="{html.escape(tooltip, quote=True)}" '
            f'title="{html.escape(tooltip, quote=True)}">'
            f'{html.escape(text[start:end])}</span>'
        )
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _timpa_text(text, offsets, masking_probs):
    masking_probs = masking_probs[:len(offsets)].float().cpu()

    pieces = []
    cursor = 0
    for offset, masking_prob in zip(offsets, masking_probs):
        start, end = offset
        if end <= start:
            continue
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))

        probability = float(masking_prob)
        strength = min(max(probability, 0.0), 1.0)
        tooltip = f"masking probability: {probability:.6g}"
        pieces.append(
            '<span class="token" '
            f'style="background-color: rgba(220, 38, 38, {strength:.3f})" '
            f'data-tooltip="{html.escape(tooltip, quote=True)}" '
            f'title="{html.escape(tooltip, quote=True)}">'
            f'{html.escape(text[start:end])}</span>'
        )
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _masked_text(text, offsets, masked_positions, mask_token):
    masked_positions = masked_positions[:len(offsets)].bool().cpu()

    pieces = []
    cursor = 0
    for offset, is_masked in zip(offsets, masked_positions):
        start, end = offset
        if end <= start:
            continue
        if start > cursor:
            pieces.append(html.escape(text[cursor:start]))

        fragment = text[start:end]
        if bool(is_masked):
            leading_space_count = len(fragment) - len(fragment.lstrip())
            leading_space = fragment[:leading_space_count]
            pieces.append(html.escape(leading_space))
            pieces.append(
                '<span class="sampled-mask" '
                f'title="masked token: {html.escape(fragment, quote=True)}">'
                f'{html.escape(mask_token)}</span>'
            )
        else:
            pieces.append(html.escape(fragment))
        cursor = max(cursor, end)
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def visualize_token_identification(
    model,
    tokenizer,
    mode,
    steers,
    texts,
    output_file="token_identification.html",
    use_chat_template=True,
):
    """Score prompt/text pairs and write an interactive token-probability HTML file."""
    mode = mode.upper()
    if mode != "AR":
        raise ValueError("mode must be 'AR'.")

    texts = [texts] if isinstance(texts, str) else texts
    if not isinstance(steers, list) or not isinstance(texts, list):
        raise TypeError("steers and texts must be lists of strings.")
    if len(steers) != len(texts):
        raise ValueError("steers and texts must contain the same number of items.")

    scores, _ = score_tokens_with_ar(
        model=model,
        tokenizer=tokenizer,
        steer=steers,
        text=texts,
        use_chat_template=use_chat_template,
    )

    encoded_texts = [
        tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        for text in texts
    ]
    cards = []
    for row, (prompt, text, encoded) in enumerate(
        zip(steers, texts, encoded_texts)
    ):
        offsets = encoded["offset_mapping"]
        token_count = len(encoded["input_ids"])
        highlighted = _highlighted_text(
            text,
            offsets,
            scores[row, :token_count],
            range(token_count),
        )
        cards.append(
            '<section class="pair">'
            '<div class="label">Prompt</div>'
            f'<div class="prompt">{html.escape(prompt)}</div>'
            '<div class="label">Text</div>'
            f'<div class="text">{highlighted}</div>'
            '</section>'
        )

    attention = "causal"
    prompt_format = "system → assistant" if use_chat_template else "raw concatenation"
    model_name = getattr(getattr(model, "config", None), "name_or_path", None)
    if not model_name:
        model_name = model.__class__.__name__
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token identification</title>
<style>
body {{ max-width: 960px; margin: 40px auto; padding: 0 20px; color: #242424;
       background: #fafafa; font: 16px/1.6 system-ui, sans-serif; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #666; margin-bottom: 28px; }}
.pair {{ background: white; border: 1px solid #ddd; border-radius: 10px;
         margin: 18px 0; padding: 20px; }}
.label {{ color: #777; font-size: 12px; font-weight: 700; letter-spacing: .08em;
          margin-top: 10px; text-transform: uppercase; }}
.prompt, .text {{ white-space: pre-wrap; }}
.token {{ border-radius: 3px; cursor: help; position: relative; }}
.token:hover::after {{
  background: #242424; border-radius: 5px; bottom: calc(100% + 7px); color: white;
  content: attr(data-tooltip); font-size: 12px; left: 50%; padding: 4px 7px;
  pointer-events: none; position: absolute; transform: translateX(-50%);
  white-space: nowrap; z-index: 10;
}}
</style>
</head>
<body>
<h1>Token identification</h1>
<div class="meta"><b>Model:</b> {html.escape(str(model_name))} ·
<b>Attention:</b> {attention} · <b>Format:</b> {prompt_format}</div>
{''.join(cards)}
</body>
</html>
"""
    output_path = Path(output_file)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def visualize_timpa(
    model,
    tokenizer,
    identifier_model,
    identifier_tokenizer,
    steer,
    text,
    use_chat_template=True,
    base_assistant_prompt="You are a helpful assistant",
    temperature=1.0,
    margin=0.05,
    refill_steps=32,
    sampling_temperature=0.0,
    refill_strategy="low_confidence",
    generator=None,
    output_file="timpa_token_identification.html",
):
    """Run probabilistic TIMPA and visualize aligned remasking probabilities."""
    texts = [text] if isinstance(text, str) else text
    if not isinstance(texts, list) or not texts:
        raise ValueError("text must be a string or a non-empty list of strings.")
    if not all(isinstance(item, str) for item in texts):
        raise TypeError("Each text must be a string.")
    if not isinstance(steer, list) or len(steer) != len(texts):
        raise ValueError("steer must contain one prompt string per text.")
    if not all(isinstance(prompt, str) for prompt in steer):
        raise TypeError("Each steer prompt must be a string.")

    tokenized_text, masking_probs, masked_positions, regenerated_texts = (
        timpa_probabilistic(
            model=model,
            tokenizer=tokenizer,
            identifier_model=identifier_model,
            identifier_tokenizer=identifier_tokenizer,
            steer=steer,
            text=texts,
            use_chat_template=use_chat_template,
            base_assistant_prompt=base_assistant_prompt,
            temperature=temperature,
            margin=margin,
            refill_steps=refill_steps,
            sampling_temperature=sampling_temperature,
            refill_strategy=refill_strategy,
            generator=generator,
        )
    )

    attention_mask = tokenized_text.get("attention_mask")
    if attention_mask is None:
        attention_mask = tokenized_text["input_ids"].new_ones(
            tokenized_text["input_ids"].shape
        )

    cards = []
    for row, (prompt, item) in enumerate(zip(steer, texts)):
        encoded = tokenizer(
            item,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        row_probs = masking_probs[row][attention_mask[row].bool()]
        row_masked_positions = masked_positions[row][attention_mask[row].bool()]
        offsets = encoded["offset_mapping"]
        if (
            row_probs.numel() != len(offsets)
            or row_masked_positions.numel() != len(offsets)
        ):
            raise RuntimeError(
                "Aligned probability or mask count does not match the diffusion "
                "token count."
            )
        highlighted = _timpa_text(
            item,
            offsets,
            row_probs,
        )
        mask_token = getattr(tokenizer, "mask_token", None) or "<|mdm_mask|>"
        sampled_text = _masked_text(
            item,
            offsets,
            row_masked_positions,
            mask_token,
        )
        cards.append(
            '<section class="card">'
            '<div class="label">Base prompt</div>'
            f'<div class="prompt">{html.escape(base_assistant_prompt)}</div>'
            '<div class="label">Steer prompt</div>'
            f'<div class="prompt">{html.escape(prompt)}</div>'
            '<div class="label">Masking probability</div>'
            f'<div class="text">{highlighted}</div>'
            '<div class="label">Masked text</div>'
            f'<div class="text">{sampled_text}</div>'
            '<div class="label">Steered text</div>'
            f'<div class="text">{html.escape(regenerated_texts[row])}</div>'
            '</section>'
        )

    prompt_format = "system → assistant" if use_chat_template else "raw concatenation"
    identifier_name = getattr(
        getattr(identifier_model, "config", None), "name_or_path", None
    ) or identifier_model.__class__.__name__
    diffusion_name = getattr(
        getattr(model, "config", None), "name_or_path", None
    ) or model.__class__.__name__
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIMPA token identification</title>
<style>
body {{ max-width: 960px; margin: 40px auto; padding: 0 20px; color: #242424;
       background: #fafafa; font: 16px/1.6 system-ui, sans-serif; }}
h1 {{ margin-bottom: 4px; }}
.meta, .legend {{ color: #666; margin-bottom: 20px; }}
.high-probability {{ color: rgb(220, 38, 38); }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px;
         margin: 18px 0; padding: 20px; }}
.label {{ color: #777; font-size: 12px; font-weight: 700; letter-spacing: .08em;
          margin-top: 10px; text-transform: uppercase; }}
.prompt, .text {{ white-space: pre-wrap; }}
.sampled-mask {{ background: #242424; border-radius: 3px; color: white;
                 padding: 1px 3px; }}
.token {{ border-radius: 3px; cursor: help; position: relative; }}
.token:hover::after {{
  background: #242424; border-radius: 5px; bottom: calc(100% + 7px); color: white;
  content: attr(data-tooltip); font-size: 12px; left: 50%; padding: 4px 7px;
  pointer-events: none; position: absolute; transform: translateX(-50%);
  white-space: nowrap; z-index: 10;
}}
</style>
</head>
<body>
<h1>TIMPA token identification</h1>
<div class="meta"><b>Identifier:</b> {html.escape(str(identifier_name))} ·
<b>Diffusion model:</b> {html.escape(str(diffusion_name))} ·
<b>Temperature:</b> {temperature:g} · <b>Margin:</b> {margin:g} ·
<b>Refill steps:</b> {refill_steps} ·
<b>Format:</b> {prompt_format}</div>
<div class="legend">More intense <span class="high-probability">red</span>
means higher masking probability.</div>
{''.join(cards)}
</body>
</html>
"""
    output_path = Path(output_file)
    output_path.write_text(document, encoding="utf-8")
    return output_path
