from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from model import MODELS
from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from utils import save_as_pdf


def _ordered_corruptions() -> list[str]:
    """All standard corruptions in group order (blur, digital, noise, weather)."""
    ordered: list[str] = []
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        ordered.extend(corruptions)
    return ordered


def compute_spearman_grid(
    metrics_df: pd.DataFrame,
    drop_dfs: dict[tuple[str, int], pd.DataFrame],
    metric: str,
    drop_col: str,
) -> pd.DataFrame:
    """Per (corruption, severity) Spearman rho between per-class cosine distance
    (`metric` in the metrics parquet) and per-class accuracy drop (`drop_col`)."""
    sub = metrics_df[metrics_df["metric"] == metric]

    corruptions = _ordered_corruptions()
    grid = pd.DataFrame(
        index=corruptions, columns=IMAGENET_C_SEVERITIES, dtype=float
    )

    for (corruption, severity), drop_df in drop_dfs.items():
        cos = (
            sub[(sub["corruption"] == corruption) & (sub["severity"] == severity)]
            .set_index("synset")["value"]
            .rename("cosine_dist")
        )
        if cos.empty:
            continue

        merged = drop_df[["synset", drop_col]].merge(cos, on="synset").dropna()
        if len(merged) < 3:
            continue

        rho = spearmanr(merged["cosine_dist"], merged[drop_col], nan_policy="omit")[0]
        if corruption in grid.index and severity in grid.columns:
            grid.loc[corruption, severity] = rho

    return grid


def render(
    grid: pd.DataFrame,
    model: str,
    metric: str,
    drop_col: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 12))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        grid.astype(float),
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman's ρ"},
    )

    ax.set_xlabel("Severity", fontsize=16)
    ax.set_ylabel("Corruption", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(
        f"Spearman(cosine distance, accuracy drop) — {MODELS.get(model, model)}",
        fontsize=16,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_as_pdf(fig, out_path)
    plt.close(fig)
