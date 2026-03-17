# visualization for linear separability
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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