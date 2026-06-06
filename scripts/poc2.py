from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

METRICS_PATH = Path("results/representations/vit_b_16_class_metrics.parquet")
OUT_DIR = Path("results/representations/taxonomy")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC = "angular_distance_median"


def signature_matrix(df, metric):
    sub = df[df["metric"] == metric].copy()
    mat = sub.pivot_table(index=["corruption", "severity"], columns="synset", values="value")
    return mat.dropna(axis=1, how="any")


def scatter_trajectories(ax, emb, ev, index, fam_color, families, title):
    groups = {}
    for (corruption, severity), (x, y) in zip(index, emb):
        groups.setdefault(corruption, []).append((severity, x, y))

    for corruption, points in groups.items():
        points.sort()
        color = fam_color[families[corruption]]
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, color=color, alpha=0.45, linewidth=1.5, zorder=2)
        for sev, x, y in points:
            ax.scatter(x, y, s=40 + sev * 25, color=color, edgecolor="black",
                       linewidth=0.6, zorder=3, alpha=0.85)
        last = points[-1]
        ax.annotate(corruption, (last[1], last[2]), fontsize=7.5, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_title(title)


def main():
    df = pd.read_parquet(METRICS_PATH)
    families = df.drop_duplicates("corruption").set_index("corruption")["group"].to_dict()

    mat = signature_matrix(df, METRIC)
    index = list(mat.index)

    fams = sorted(set(families.values()))
    cmap = plt.get_cmap("tab10")
    fam_color = {f: cmap(i) for i, f in enumerate(fams)}

    raw = mat.values
    resid = raw - raw.mean(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, data, name in [(axes[0], raw, "raw"), (axes[1], resid, "residual (class-centered)")]:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(data)
        scatter_trajectories(ax, emb, pca.explained_variance_ratio_,
                             index, fam_color, families,
                             f"{METRIC} all severities — {name}")

    family_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=fam_color[f], label=f)
        for f in fams
    ]
    sev_handles = [
        plt.scatter([], [], s=40 + sev * 25, color="gray", edgecolor="black", label=f"sev {sev}")
        for sev in range(1, 6)
    ]
    fig.legend(handles=family_handles + sev_handles,
               loc="upper center", ncol=len(fams) + 5, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / f"corruption_map_{METRIC}_all_sev.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"corruptions={len(set(c for c, _ in index))}  "
          f"severities=5  classes={mat.shape[1]}")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()