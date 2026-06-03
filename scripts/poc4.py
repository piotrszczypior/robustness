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

METRICS = ["angular_distance_median"]
K = 4
N_BOOT = 500
SEED = 0


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
    x = mat.values if hasattr(mat, "values") else mat
    ranked = pd.DataFrame(x).rank(axis=1).values
    return np.corrcoef(ranked)


def cluster(sim):
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    return Z, order


def labels_from_sim(sim, k):
    Z, _ = cluster(sim)
    return fcluster(Z, t=k, criterion="maxclust")


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


def bootstrap_coassoc(mat, k, n_boot, seed):
    x = mat.values if hasattr(mat, "values") else mat
    rng = np.random.default_rng(seed)
    n_cond, n_cls = x.shape
    coassoc = np.zeros((n_cond, n_cond))
    for _ in range(n_boot):
        idx = rng.integers(0, n_cls, n_cls)
        sim_b = spearman_matrix(x[:, idx])
        lab = labels_from_sim(sim_b, k)
        coassoc += (lab[:, None] == lab[None, :]).astype(float)
    return coassoc / n_boot


def cluster_stability(coassoc, ref_labels):
    ref = np.asarray(ref_labels)
    out = {}
    for cid in sorted(set(ref.tolist())):
        members = np.where(ref == cid)[0]
        if len(members) > 1:
            sub = coassoc[np.ix_(members, members)]
            iu = np.triu_indices(len(members), k=1)
            out[cid] = float(sub[iu].mean())
        else:
            out[cid] = 1.0
    return out


def heatmap(sim, labels, order, title, path, tag, cmap, vmin, vmax):
    font_size = 10 if tag == "15x15" else 6
    s = sim[np.ix_(order, order)]
    lab = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(s, cmap=cmap, vmin=vmin, vmax=vmax)
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
        for name, base in [("raw", mat), ("residual", mat - mat.mean(axis=0))]:
            sim = spearman_matrix(base)
            Z, order = cluster(sim)
            clabels, ari, nmi = evaluate(Z, groups, K)
            print(f"  [{tag}] {name:8s} ARI={ari:.3f}  NMI={nmi:.3f}")
            heatmap(sim, labels, order, f"{metric} {tag} {name}", OUT_DIR / f"{metric}_{tag}_{name}_heatmap.png", tag, "RdBu_r", -1.0, 1.0)
            dendro(Z, labels, f"{metric} {tag} {name}", OUT_DIR / f"{metric}_{tag}_{name}_dendro.png", tag)
            coassoc = bootstrap_coassoc(base, K, N_BOOT, SEED)
            stab = cluster_stability(coassoc, clabels)
            stab_str = "  ".join(f"c{cid}={v:.2f}" for cid, v in stab.items())
            print(f"  [{tag}] {name:8s} stability  {stab_str}")
            heatmap(coassoc, labels, order, f"{metric} {tag} {name} coassoc", OUT_DIR / f"{metric}_{tag}_{name}_coassoc.png", tag, "viridis", 0.0, 1.0)
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