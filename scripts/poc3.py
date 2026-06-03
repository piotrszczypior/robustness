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
OUT_DIR = Path("results/representations/taxonomy")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS = ["relative_shift_median", "angular_distance_median", "tangential_fraction_median"]
K = 4


def signature(df, metric):
    sub = df[df["metric"] == metric]
    mat = sub.pivot_table(index=["corruption", "severity"], columns="synset", values="value")
    mat = mat.dropna(axis=1, how="any")
    labels = [f"{c}_s{s}" for c, s in mat.index]
    corr = [c for c, _ in mat.index]
    return mat, labels, corr


def spearman_sim(mat, residual=True):
    x = mat.values.astype(float)
    if residual:
        x = x - x.mean(axis=0, keepdims=True)
    ranks = pd.DataFrame(x).rank(axis=1).values
    return np.corrcoef(ranks)


def cluster(sim):
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    return Z, order


def heatmap(sim, labels, order, title, path):
    s = sim[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(s, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)))
    ax.set_yticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=90, fontsize=5)
    ax.set_yticklabels(lab, fontsize=5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dendro(Z, labels, title, path):
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=5)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_parquet(METRICS_PATH)
    fam = df.drop_duplicates("corruption").set_index("corruption")["group"].to_dict()

    sims, labels, corr = {}, None, None
    for m in METRICS:
        mat, labels, corr = signature(df, m)
        sims[m] = spearman_sim(mat, residual=True)
    families = [fam[c] for c in corr]

    per_metric_labels = {}
    for m in METRICS:
        Z, _ = cluster(sims[m])
        per_metric_labels[m] = fcluster(Z, t=K, criterion="maxclust")

    print("zgodnosc miedzy metrykami (ARI):")
    for i in range(len(METRICS)):
        for j in range(i + 1, len(METRICS)):
            a = adjusted_rand_score(per_metric_labels[METRICS[i]], per_metric_labels[METRICS[j]])
            print(f"  {METRICS[i]:26s} vs {METRICS[j]:26s} {a:.3f}")

    print("\nkazda metryka vs Hendrycks:")
    for m in METRICS:
        a = adjusted_rand_score(families, per_metric_labels[m])
        n = normalized_mutual_info_score(families, per_metric_labels[m])
        print(f"  {m:26s} ARI={a:.3f} NMI={n:.3f}")

    fused = 1.0 - np.mean([1.0 - sims[m] for m in METRICS], axis=0)
    Zf, orderf = cluster(fused)
    fused_cl = fcluster(Zf, t=K, criterion="maxclust")
    print("\nfuzja 3 metryk vs Hendrycks:")
    print(f"  ARI={adjusted_rand_score(families, fused_cl):.3f} NMI={normalized_mutual_info_score(families, fused_cl):.3f}")

    heatmap(fused, labels, orderf, "fused 3-metric (residual)", OUT_DIR / "fused_heatmap.png")
    dendro(Zf, labels, "fused 3-metric (residual)", OUT_DIR / "fused_dendro.png")

    print("\nklastry fuzji:")
    for cid in sorted(set(fused_cl)):
        members = sorted({corr[i] for i in range(len(corr)) if fused_cl[i] == cid})
        print(f"  cluster {cid}: {members}")

    print(f"\nfigures -> {OUT_DIR}")


if __name__ == "__main__":
    main()