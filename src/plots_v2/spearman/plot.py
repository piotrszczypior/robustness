from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr


def compute_spearman(dfs: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    vectors = pd.DataFrame(
        {name: df.set_index("synset")[metric] for name, df in dfs.items()}
    ).dropna()

    corr, _ = spearmanr(vectors.values, axis=0)

    labels = list(dfs.keys())
    if len(labels) == 1:
        corr = np.array([[1.0]])

    return pd.DataFrame(corr, index=labels, columns=labels)


def render(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    bounds = np.arange(0.4, 1.05, 0.05)
    base_cmap = plt.get_cmap("coolwarm")
    cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, len(bounds) - 1)))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        vmin=0.3,
        vmax=0.9,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman's Rank Correlation (ρ)", "alpha": 0.55},
    )

    ax.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=True,
    )
    plt.xticks(rotation=60, ha="left")

    fig.tight_layout()
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, out_path)
    plt.close(fig)
