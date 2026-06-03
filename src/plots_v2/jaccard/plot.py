from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile
from fragile.definitions import FragileDefinition


def compute_jaccard(dfs: dict[str, pd.DataFrame], definition: FragileDefinition) -> pd.DataFrame:
    fragile_sets: dict[str, set] = {}
    for name, df in dfs.items():
        df = get_absolute_fragile(df)
        df = get_relative_drop_fragile(df)
        mask = definition.combine(df)
        fragile_sets[name] = set(df.loc[mask, "synset"])

    labels = list(dfs.keys())
    n = len(labels)
    matrix = np.zeros((n, n))

    for i, name_i in enumerate(labels):
        for j, name_j in enumerate(labels):
            set_i = fragile_sets[name_i]
            set_j = fragile_sets[name_j]
            union = len(set_i | set_j)
            matrix[i, j] = len(set_i & set_j) / union if union > 0 else 1.0

    return pd.DataFrame(matrix, index=labels, columns=labels)


def render(jaccard: pd.DataFrame, definition: FragileDefinition, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        jaccard,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Jaccard Index ({definition.label})", "alpha": 0.55},
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
