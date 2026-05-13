# visualization for linear separability
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import html as html_lib

def viz_separability(pos_hiddens, neg_hiddens):
    n_layers = len(pos_hiddens)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.8, n_rows*3.2), squeeze=False)
    fig.suptitle("PCA 2D projection per layer — pos vs neg hidden states", y=0.98)

    for i in range(n_layers):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        pos = pos_hiddens[i]  # [500, 4096]
        neg = neg_hiddens[i]  # [500, 4096]

        # combine: [1000, 4096]
        X = torch.cat([pos, neg], dim=0).detach().float().cpu().numpy()

        # PCA -> [1000, 2]
        Z = PCA(n_components=2).fit_transform(X)

        n_pos = pos.shape[0]
        ax.scatter(Z[:n_pos, 0], Z[:n_pos, 1], s=6, alpha=0.7, label="pos")
        ax.scatter(Z[n_pos:, 0], Z[n_pos:, 1], s=6, alpha=0.7, label="neg")

        ax.set_title(f"layer {i}")
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off any unused axes
    for j in range(n_layers, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    # only one legend (optional)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)

    plt.tight_layout()
    plt.show()


def clean_token(tok):
    return tok.replace("Ġ", "").replace("▁", "")

def visualize_token_mask(
    steered_x,
    tokenizer,
    path="highlighted.html",
):
    all_parts = []

    for steer_step in range(len(steered_x)):
        input_ids = steered_x[steer_step]["before"]
        after_steered = steered_x[steer_step]["after"]
        mask_probs = steered_x[steer_step]["mask_probs"]
        mask = steered_x[steer_step]["steer_mask"]

        B, T = input_ids.shape

        phase_parts = [f"""
        <section class="phase">
            <h1>Phase {steer_step + 1}</h1>
        """]

        for b in range(B):
            ids = input_ids[b].tolist()

            raw_tokens = tokenizer.convert_ids_to_tokens(
                ids,
                skip_special_tokens=True,
            )

            tokens = [
                tokenizer.convert_tokens_to_string([t])
                for t in raw_tokens
            ]

            non_special_tokens = [
                i for i, tok_id in enumerate(ids)
                if tok_id not in tokenizer.all_special_ids
            ]

            updated_mask = mask[b][non_special_tokens]
            updated_probs = mask_probs[b][non_special_tokens]

            sentence_html = "".join(
                (
                    f'<span class="highlight" '
                    f'style="--alpha:{float(prob):.3f}" '
                    f'data-prob="{float(prob):.3f}">'
                    f'{html_lib.escape(tok)}'
                    f'</span>'
                )
                if keep
                else html_lib.escape(tok)
                for tok, keep, prob in zip(
                    tokens,
                    updated_mask.tolist(),
                    updated_probs.tolist(),
                )
            )

            phase_parts.append(f"""
            <section class="sentence">
                <h2>Sentence {b + 1}</h2>
                <p>{sentence_html}</p>
            </section>
            """)

        phase_parts.append("</section>")
        all_parts.append("\n".join(phase_parts))

    final_output = after_steered
    final_parts = ["""
    <section class="phase final">
        <h1>Final Text</h1>
    """]

    for b in range(final_output.shape[0]):
        ids = final_output[b].tolist()

        raw_tokens = tokenizer.convert_ids_to_tokens(
            ids,
            skip_special_tokens=True,
        )

        tokens = [
            tokenizer.convert_tokens_to_string([t])
            for t in raw_tokens
        ]

        final_text = "".join(html_lib.escape(tok) for tok in tokens)

        final_parts.append(f"""
        <section class="sentence">
            <h2>Sentence {b + 1}</h2>
            <p>{final_text}</p>
        </section>
        """)

    final_parts.append("</section>")
    all_parts.append("\n".join(final_parts))

    html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: sans-serif;
            line-height: 1.7;
            font-size: 18px;
            padding: 24px;
        }}

        .phase {{
            margin-bottom: 44px;
            padding-bottom: 28px;
            border-bottom: 2px solid #ddd;
        }}

        .phase h1 {{
            font-size: 28px;
            margin-bottom: 20px;
        }}

        .sentence {{
            margin-bottom: 28px;
        }}

        .sentence h2 {{
            font-size: 20px;
            margin-bottom: 8px;
        }}

        .highlight {{
            position: relative;
            background-color: rgba(250, 187, 142, var(--alpha));
            border-radius: 3px;
            padding: 1px 2px;
        }}

        .highlight:hover {{
            background-color: rgba(255, 84, 121, 1.0);
        }}

        .highlight:hover::after {{
            content: attr(data-prob);
            position: absolute;
            left: 0;
            bottom: 100%;
            background: #222;
            color: white;
            font-size: 12px;
            padding: 2px 5px;
            border-radius: 4px;
            white-space: nowrap;
            z-index: 10;
        }}

    </style>
    </head>
    <body>
    {''.join(all_parts)}
    </body>
    </html>
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)