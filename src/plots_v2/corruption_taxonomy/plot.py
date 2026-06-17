from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from utils import save_as_pdf


def load_signatures(df: pd.DataFrame, metric: str, aggregate_severity: bool):
    sub = df[df["metric"] == metric]
    if aggregate_severity:
        mat = sub.pivot_table(index="corruption", columns="synset", values="value", aggfunc="mean")
        labels = list(mat.index)
    else:
        mat = sub.pivot_table(index=["corruption", "severity"], columns="synset", values="value")
        labels = [f"{c}_s{s}" for c, s in mat.index]
    mat = mat.dropna(axis=1, how="any")
    return mat, labels


def spearman_matrix(mat: pd.DataFrame) -> np.ndarray:
    ranked = mat.rank(axis=1).values
    return np.corrcoef(ranked)


def cluster(sim: np.ndarray) -> np.ndarray:
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="average")


def heatmap(sim: np.ndarray, labels: list[str], title: str, path: Path, tag: str) -> None:
    font_size = 10 if tag == "15x15" else 6
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=font_size)
    ax.set_yticklabels(labels, fontsize=font_size)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_as_pdf(fig, path)
    plt.close(fig)


def dendro(Z: np.ndarray, labels: list[str], title: str, path: Path, tag: str) -> None:
    font_size = 9 if tag == "15x15" else 6
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=font_size)
    ax.set_title(title)
    fig.tight_layout()
    save_as_pdf(fig, path)
    plt.close(fig)


def render(df: pd.DataFrame, metric: str, out_base: Path) -> None:
    out_dir = out_base / "images" / "v3" / "corruption_taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for aggregate, tag in [(False, "75x75"), (True, "15x15")]:
        mat, labels = load_signatures(df, metric, aggregate)
        sim = spearman_matrix(mat)
        Z = cluster(sim)

        heatmap(
            sim, labels,
            "Spearman rank correlation of corruption groups",
            out_dir / f"{metric}_{tag}_heatmap.png",
            tag,
        )
        dendro(
            Z, labels,
            "Dendrogram of hierarchical clustering of corruptions",
            out_dir / f"{metric}_{tag}_dendro.png",
            tag,
        )
        print(f"  {metric}_{tag}_heatmap.pdf, {metric}_{tag}_dendro.pdf")
