from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

MODELS = [
    ("resnet50", Path("results/representations/resnet50_class_metrics.parquet")),
    ("vit_b_16", Path("results/representations/vit_b_16_class_metrics.parquet")),
    # ("convnext_base", Path("results/representations/convnext_base_class_metrics.parquet")),
]
METRIC = "angular_distance_median"
K = 4
N_PERM = 9999
SEED = 0


def common_corruptions(models):
    sets = [set(pd.read_parquet(p, columns=["corruption"])["corruption"].unique()) for _, p in models]
    return sorted(set.intersection(*sets))


def signature_similarity(path, metric, corruptions):
    df = pd.read_parquet(path)
    sub = df[df["metric"] == metric]
    mat = sub.pivot_table(index="corruption", columns="synset", values="value", aggfunc="mean")
    mat = mat.reindex(corruptions).dropna(axis=1, how="any")
    ranked = mat.rank(axis=1).values
    return np.corrcoef(ranked)


def partition(sim, k):
    dist = 1.0 - sim
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(Z, t=k, criterion="maxclust")


def upper(m):
    return m[np.triu_indices_from(m, k=1)]


def scorr(a, b):
    return spearmanr(a, b)[0]


def mantel(A, B, n_perm, rng):
    a = upper(A)
    obs = scorr(a, upper(B))
    n = A.shape[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if scorr(a, upper(B[p][:, p])) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    corruptions = common_corruptions(MODELS)
    print(f"metric={METRIC}  corruptions aligned (n={len(corruptions)})")

    sims, parts = {}, {}
    for name, path in MODELS:
        s = signature_similarity(path, METRIC, corruptions)
        sims[name] = s
        parts[name] = partition(s, K)

    rng = np.random.default_rng(SEED)
    names = [n for n, _ in MODELS]
    print("\ncross-architecture comparison:")
    print(f"  {'pair':32s} mantel_r   p        ARI(k=4)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, p = mantel(sims[names[i]], sims[names[j]], N_PERM, rng)
            ari = adjusted_rand_score(parts[names[i]], parts[names[j]])
            pair = f"{names[i]} vs {names[j]}"
            print(f"  {pair:32s} {r:+.3f}    {p:.4f}   {ari:.3f}")


if __name__ == "__main__":
    main()