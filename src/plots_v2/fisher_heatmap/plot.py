from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import fisher_exact

from model import MODELS
from fragile.definitions import DEFINITIONS
from fragile.fragile import (
    get_absolute_fragile,
    get_relative_drop_fragile,
    get_rmce_fragile,
    get_strongly_fragile,
)
from fragile.methods import calculate_relative_drop


def _get_fragile(df: pd.DataFrame, alexnet_df: pd.DataFrame, definition) -> pd.DataFrame:
    df = calculate_relative_drop(df)
    df_a = get_absolute_fragile(df)
    df_b = get_relative_drop_fragile(df)
    df_c = get_rmce_fragile(df, alexnet_df)
    strong_fragile = get_strongly_fragile(df_a, df_b, df_c, definition)
    return df.merge(strong_fragile, on="synset")


def compute_fisher_matrix(
    dfs: dict[str, pd.DataFrame],
    definition_name: str = "ab",
) -> pd.DataFrame:
    definition = DEFINITIONS[definition_name]
    model_keys = [k for k in MODELS.keys() if k in dfs]
    n_models = len(model_keys)

    fragile_vectors: dict[str, np.ndarray] = {}
    for model in model_keys:
        df = _get_fragile(dfs[model], dfs["alexnet"], definition)
        df_sorted = df.sort_values("synset")
        fragile_vectors[model] = df_sorted["is_strongly_fragile"].values.astype(bool)

    p_matrix = np.ones((n_models, n_models))

    for i, model_i in enumerate(model_keys):
        for j, model_j in enumerate(model_keys):
            if i == j:
                continue
            vi = fragile_vectors[model_i]
            vj = fragile_vectors[model_j]

            a = np.sum(vi & vj)
            b = np.sum(vi & ~vj)
            c = np.sum(~vi & vj)
            d = np.sum(~vi & ~vj)

            contingency = [[a, b], [c, d]]
            _, p_value = fisher_exact(contingency)
            p_matrix[i, j] = p_value

    labels = [MODELS[k] for k in model_keys]
    return pd.DataFrame(p_matrix, index=labels, columns=labels)


def render(p_matrix: pd.DataFrame, out_path: Path, title: str = "") -> None:
    log_p = -np.log10(p_matrix.values + 1e-100)
    np.fill_diagonal(log_p, 0)

    log_df = pd.DataFrame(log_p, index=p_matrix.index, columns=p_matrix.columns)

    fig, ax = plt.subplots(figsize=(14, 10))

    vmax = min(80, np.max(log_p[log_p < np.inf]))

    sns.heatmap(
        log_df,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "-log10(p-value)"},
        mask=np.eye(len(log_df), dtype=bool),
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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
