from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

from fragile.fragile import get_rmce_fragile
from utils import save_as_pdf

from plots_v2.fisher_heatmap.plot import render as render_fisher


def compute_c_sets(
    raw_dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, set], dict[str, set]]:
    """Build the criterion-C (RmCE) fragile set per model.

    Returns (c_sets, universes) keyed by model key. `c_sets[m]` are the synsets
    flagged by criterion C; `universes[m]` are all synsets evaluated for that model.
    """
    alexnet_df = raw_dfs["alexnet"]

    c_sets: dict[str, set] = {}
    universes: dict[str, set] = {}
    for model, df in raw_dfs.items():
        scored = get_rmce_fragile(df, alexnet_df)
        c_sets[model] = set(scored.loc[scored["is_fragile_c"] == 1, "synset"])
        universes[model] = set(scored["synset"])
    return c_sets, universes


def compute_jaccard_matrix(
    c_sets: dict[str, set], labels: list[str]
) -> pd.DataFrame:
    n = len(labels)
    matrix = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            si, sj = c_sets[li], c_sets[lj]
            union = len(si | sj)
            matrix[i, j] = len(si & sj) / union if union > 0 else 1.0
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_fisher_matrix(
    c_sets: dict[str, set], universes: dict[str, set], labels: list[str]
) -> pd.DataFrame:
    n = len(labels)
    p_matrix = np.ones((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            universe = universes[li] & universes[lj]
            ci = c_sets[li] & universe
            cj = c_sets[lj] & universe

            a = len(ci & cj)
            b = len(ci - cj)
            c = len(cj - ci)
            d = len(universe) - a - b - c

            _, p_value = fisher_exact([[a, b], [c, d]])
            p_matrix[i, j] = p_value
    return pd.DataFrame(p_matrix, index=labels, columns=labels)


def render_jaccard(jaccard: pd.DataFrame, out_path: Path, title: str = "") -> None:
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
        cbar_kws={"label": "Jaccard Index (criterion C)", "alpha": 0.55},
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

    if title:
        ax.set_title(title, fontsize=14, pad=20)

    fig.tight_layout()
    save_as_pdf(fig, out_path)
    plt.close(fig)
