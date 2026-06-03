from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

METRICS_PATH = Path("results/representations/vit_b_16_class_metrics.parquet")
OUT_DIR = Path("representations/taxonomy2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["angular_distance_median"]
K = 4


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
    order = dendrogram(Z, no_plot=True)["leaves"]
    return Z, order


def heatmap(sim, labels, order, title, path, tag):
    font_size = 10 if tag == "15x15" else 6
    # s = sim[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(lab)))
    ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=90, fontsize=font_size)
    ax.set_yticklabels(lab, fontsize=font_size)
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


def print_clusters(labels, clabels):
    for cid in sorted(set(clabels)):
        members = [labels[i] for i in range(len(labels)) if clabels[i] == cid]
        print(f"    cluster {cid}: {sorted(members)}")


def analyze(df, metric):
    print(f"\n==== {metric} ====")
    for aggregate, tag in [(False, "75x75"), (True, "15x15")]:
        mat, labels = load_signatures(df, metric, aggregate)
        sim = spearman_matrix(mat)
        Z, order = cluster(sim)
        heatmap(sim, labels, order, f"{metric} {tag}", OUT_DIR / f"{metric}_{tag}_vit_b_16_heatmap.png", tag)
        dendro(Z, labels, f"{metric} {tag}", OUT_DIR / f"{metric}_{tag}_vit_b_16_dendro.png", tag)
        if tag == "15x15":
            clabels = fcluster(Z, t=K, criterion="maxclust")
            print_clusters(labels, clabels)


def main():
    df = pd.read_parquet(METRICS_PATH)
    severities = [3]
    df = df[df["severity"].isin(severities)]

    for metric in METRICS:
        analyze(df, metric)
    print(f"\nfigures -> {OUT_DIR}")


if __name__ == "__main__":
    main()