import html
from pathlib import Path

from timpateks import score_tokens_wrt_steer


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
    if mode not in {"AR", "DLM"}:
        raise ValueError("mode must be either 'AR' or 'DLM'.")

    texts = [texts] if isinstance(texts, str) else texts
    if not isinstance(steers, list) or not isinstance(texts, list):
        raise TypeError("steers and texts must be lists of strings.")
    if len(steers) != len(texts):
        raise ValueError("steers and texts must contain the same number of items.")

    scores, text_token_indices = score_tokens_wrt_steer(
        model=model,
        tokenizer=tokenizer,
        steer=steers,
        text=texts,
        identifier_mode=mode,
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
            text_token_indices[row, :token_count].cpu(),
        )
        cards.append(
            '<section class="pair">'
            '<div class="label">Prompt</div>'
            f'<div class="prompt">{html.escape(prompt)}</div>'
            '<div class="label">Text</div>'
            f'<div class="text">{highlighted}</div>'
            '</section>'
        )

    attention = "causal" if mode == "AR" else "bidirectional"
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


def visualize_token_identification_comparison(
    model,
    tokenizer,
    mode,
    steers,
    text,
    output_file="token_identification_comparison.html",
    use_chat_template=True,
):
    """Compare per-token probabilities for one text under two steer prompts."""
    mode = mode.upper()
    if mode not in {"AR", "DLM"}:
        raise ValueError("mode must be either 'AR' or 'DLM'.")
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(steers, list) or len(steers) != 2:
        raise ValueError("steers must contain exactly two prompt strings.")
    if not all(isinstance(steer, str) for steer in steers):
        raise TypeError("Each steer must be a string.")

    scores, _ = score_tokens_wrt_steer(
        model=model,
        tokenizer=tokenizer,
        steer=steers,
        text=[text, text],
        identifier_mode=mode,
        use_chat_template=use_chat_template,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_count = len(encoded["input_ids"])
    highlighted = _comparison_text(
        text,
        encoded["offset_mapping"],
        scores[0, :token_count],
        scores[1, :token_count],
    )

    attention = "causal" if mode == "AR" else "bidirectional"
    prompt_format = "system → assistant" if use_chat_template else "raw concatenation"
    model_name = getattr(getattr(model, "config", None), "name_or_path", None)
    if not model_name:
        model_name = model.__class__.__name__
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token identification comparison</title>
<style>
body {{ max-width: 960px; margin: 40px auto; padding: 0 20px; color: #242424;
       background: #fafafa; font: 16px/1.6 system-ui, sans-serif; }}
h1 {{ margin-bottom: 4px; }}
.meta, .legend {{ color: #666; margin-bottom: 20px; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px;
         margin: 18px 0; padding: 20px; }}
.label {{ color: #777; font-size: 12px; font-weight: 700; letter-spacing: .08em;
          margin-top: 10px; text-transform: uppercase; }}
.prompt, .text {{ white-space: pre-wrap; }}
.positive {{ color: rgb(37, 99, 235); }}
.negative {{ color: rgb(220, 38, 38); }}
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
<h1>Token identification comparison</h1>
<div class="meta"><b>Model:</b> {html.escape(str(model_name))} ·
<b>Attention:</b> {attention} · <b>Format:</b> {prompt_format}</div>
<div class="legend">Difference = prompt 2 − prompt 1 ·
<span class="positive">blue: higher</span> ·
<span class="negative">red: lower</span></div>
<section class="card">
<div class="label">Prompt 1</div>
<div class="prompt">{html.escape(steers[0])}</div>
<div class="label">Prompt 2</div>
<div class="prompt">{html.escape(steers[1])}</div>
<div class="label">Token probability difference</div>
<div class="text">{highlighted}</div>
</section>
</body>
</html>
"""
    output_path = Path(output_file)
    output_path.write_text(document, encoding="utf-8")
    return output_path
