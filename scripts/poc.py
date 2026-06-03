from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

METRICS_PATH = Path("results/representations/resnet50_class_metrics.parquet")
OUT_DIR = Path("representations/taxonomy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS =["angular_distance_median"]
K = 4


def load_signatures(df, group_map, metric, aggregate_severity):
    sub = df[df["metric"] == metric]
    if aggregate_severity:
        mat = sub.pivot_table(index="corruption", columns="synset", values="value", aggfunc="mean")
        labels = list(mat.index)
        groups = [group_map[c] for c in labels]
    else:
        mat = sub.pivot_table(index=["corruption", "severity"], columns="synset", values="value")
        labels = [f"{c}_s{s}" for c, s in mat.index]
        groups = [group_map[c] for c, _ in mat.index]
    mat = mat.dropna(axis=1, how="any")
    return mat, labels, groups


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


def evaluate(Z, groups, k):
    labels = fcluster(Z, t=k, criterion="maxclust")
    ari = adjusted_rand_score(groups, labels)
    nmi = normalized_mutual_info_score(groups, labels)
    return labels, ari, nmi


def variance_decomposition(mat):
    x = mat.values
    grand = x.mean()
    class_mean = x.mean(axis=0, keepdims=True)
    cond_mean = x.mean(axis=1, keepdims=True)
    resid = x - class_mean - cond_mean + grand
    ss_total = ((x - grand) ** 2).sum()
    ss_class = ((class_mean - grand) ** 2).sum() * x.shape[0]
    ss_cond = ((cond_mean - grand) ** 2).sum() * x.shape[1]
    ss_resid = (resid ** 2).sum()
    return ss_class / ss_total, ss_cond / ss_total, ss_resid / ss_total


def heatmap(sim, labels, order, title, path, tag):
    font_size = 10 if tag == "15x15" else 6 

    s = sim[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(s, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
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


def analyze(df, group_map, metric):
    print(f"\n==== {metric} ====")
    for aggregate, tag in [(False, "75x75"), (True, "15x15")]:
        mat, labels, groups = load_signatures(df, group_map, metric, aggregate)
        fc, fco, fr = variance_decomposition(mat)
        print(f"  [{tag}] variance  class={fc:.3f}  condition={fco:.3f}  residual={fr:.3f}")
        raw = spearman_matrix(mat)
        resid = spearman_matrix(mat - mat.mean(axis=0))
        for name, sim in [("raw", raw), ("residual", resid)]:
            Z, order = cluster(sim)
            clabels, ari, nmi = evaluate(Z, groups, K)
            print(f"  [{tag}] {name:8s} ARI={ari:.3f}  NMI={nmi:.3f}")
            heatmap(sim, labels, order, f"Spearman rank correlation of corruption groups", OUT_DIR / f"{metric}_{tag}_{name}_heatmap.png", tag)
            dendro(Z, labels, f"Dendogram of hierarchical clustering of curruption for ResNet-50", OUT_DIR / f"{metric}_{tag}_{name}_dendro.png", tag)
            if tag == "15x15":
                print_clusters(labels, clabels)


def main():
    df = pd.read_parquet(METRICS_PATH)
    group_map = df.drop_duplicates("corruption").set_index("corruption")["group"].to_dict()
    for metric in METRICS:
        analyze(df, group_map, metric)
    print(f"\nfigures -> {OUT_DIR}")


if __name__ == "__main__":
    main()