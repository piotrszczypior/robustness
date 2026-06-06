from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

METRICS_PATH = Path("results/representations/vit_b_16_class_metrics.parquet")
OUT_DIR = Path("representations/taxonomy4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["angular_distance_median"]


def load_signatures(df, metric, aggregate_severity):
    sub = df[df["metric"] == metric]
    if aggregate_severity:
        mat = sub.pivot_table(index="corruption", columns="synset", values="value", aggfunc="mean")
        labels = list(mat.index)
    else:
        mat = sub.pivot_table(index=["corruption", "severity"], columns="synset", values="value")
        labels = [f"{c}_s{s}" for c, s in mat.index]
    mat = mat.dropna(axis=1, how="any")
    return mat, labels


def spearman_matrix(mat):
    ranked = mat.rank(axis=1).values
    return np.corrcoef(ranked)


def cluster(sim):
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return Z

def heatmap(sim, labels, title, path, tag):
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dendro(Z, labels, title, path, tag):
    font_size = 9 if tag == "15x15" else 6
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=font_size)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(df, metric):
    for aggregate, tag in [(False, "75x75"), (True, "15x15")]:
        mat, labels = load_signatures(df, metric, aggregate)
        sim = spearman_matrix(mat)
        Z = cluster(sim)
        heatmap(sim, labels, "Spearman rank correlation of corruption groups", OUT_DIR / f"cosine_distance_{tag}_heatmap.png", tag)
        dendro(Z, labels, "Dendrogram of hierarchical clustering of corruptions for ViT-B/16", OUT_DIR / f"cosine_distance_{tag}_dendro.png", tag)


def main():
    df = pd.read_parquet(METRICS_PATH)
    for metric in METRICS:
        analyze(df, metric)
    
    print(f"figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()