from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan

FEATURES = ["acc_clean", "acc_corrupt", "rel_drop", "abs_drop", "RmCE", "mCE"]
Projection = Literal["pca", "umap"]


def run_clustering(df: pd.DataFrame, projection: Projection = "umap") -> None:
    df = run_hdbscan(df)
    df = run_projection(df, projection=projection)
    plot_clustering(df, projection=projection)


def run_hdbscan(df: pd.DataFrame, min_cluster_size: int = 8) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        cluster_selection_epsilon=0.1,
        gen_min_span_tree=True,
    )
    df.loc[X.index, "cluster"] = clusterer.fit_predict(X_scaled)
    df.loc[X.index, "cluster_prob"] = clusterer.probabilities_
    return df


def run_projection(df: pd.DataFrame, projection: Projection = "pca") -> pd.DataFrame:
    if projection == "pca":
        return run_pca(df)
    return run_umap(df)


def run_pca(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df.loc[X.index, "dim1"] = components[:, 0]
    df.loc[X.index, "dim2"] = components[:, 1]
    df.attrs["explained_variance"] = pca.explained_variance_ratio_
    df.attrs["projection"] = "pca"
    return df


def run_umap(df: pd.DataFrame) -> pd.DataFrame:
    import umap
    df = df.copy()
    X = df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(n_components=2, random_state=42)
    components = reducer.fit_transform(X_scaled)
    df.loc[X.index, "dim1"] = components[:, 0]
    df.loc[X.index, "dim2"] = components[:, 1]
    df.attrs["projection"] = "umap"
    return df


def plot_clustering(
    df: pd.DataFrame,
    projection: Projection = "pca",
    output_path: str = "images/fragile/clustering",
    fragile_threshold: int = 15,
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    proj = df.attrs.get("projection", projection).upper()
    ev = df.attrs.get("explained_variance", [0, 0])
    x_label = f"{proj}1 ({ev[0]:.1%})" if projection == "pca" else f"{proj}1"
    y_label = f"{proj}2 ({ev[1]:.1%})" if projection == "pca" else f"{proj}2"

    fig_cluster, ax_cluster = plt.subplots(figsize=(7, 6))
    clusters = df["cluster"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    color_map = {c: colors[i] for i, c in enumerate(sorted(clusters))}
    color_map[-1] = (0.5, 0.5, 0.5, 0.3)
    for cluster, group in df.groupby("cluster"):
        ax_cluster.scatter(
            group["dim1"], group["dim2"],
            c=[color_map[cluster]],
            label=f"cluster {int(cluster)}" if cluster != -1 else "noise",
            alpha=0.6, s=15,
        )
    ax_cluster.set_title(f"{proj} colored by HDBSCAN cluster")
    ax_cluster.set_xlabel(x_label)
    ax_cluster.set_ylabel(y_label)
    ax_cluster.legend(fontsize=8, frameon=False)
    fig_cluster.tight_layout()
    fig_cluster.savefig(output_dir / f"{projection}_hdbscan_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cluster)

    fig_fragile, ax_fragile = plt.subplots(figsize=(7, 6))
    sc = ax_fragile.scatter(
        df["dim1"], df["dim2"],
        c=df["fragile_count"], cmap="RdYlBu_r", alpha=0.6, s=15,
    )
    plt.colorbar(sc, ax=ax_fragile, label="fragile_count")
    ax_fragile.set_title(f"{proj} colored by fragile_count")
    ax_fragile.set_xlabel(x_label)
    ax_fragile.set_ylabel(y_label)
    fig_fragile.tight_layout()
    fig_fragile.savefig(output_dir / f"{projection}_hdbscan_fragile_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig_fragile)

    fragile_mask = df["fragile_count"] >= fragile_threshold
    fig_highlight, ax_highlight = plt.subplots(figsize=(7, 6))
    ax_highlight.scatter(
        df[~fragile_mask]["dim1"], df[~fragile_mask]["dim2"],
        c="lightgray", alpha=0.4, s=12, label="not fragile",
    )
    sc2 = ax_highlight.scatter(
        df[fragile_mask]["dim1"], df[fragile_mask]["dim2"],
        c=df[fragile_mask]["fragile_count"], cmap="Reds",
        alpha=0.9, s=40, zorder=5, label=f"fragile (≥{fragile_threshold})",
    )
    plt.colorbar(sc2, ax=ax_highlight, label="fragile_count")
    ax_highlight.set_title(f"{proj} — fragile classes highlighted (threshold={fragile_threshold})")
    ax_highlight.set_xlabel(x_label)
    ax_highlight.set_ylabel(y_label)
    ax_highlight.legend(fontsize=8, frameon=False)
    fig_highlight.tight_layout()
    fig_highlight.savefig(output_dir / f"{projection}_fragile_highlighted.png", dpi=150, bbox_inches="tight")
    plt.close(fig_highlight)

    print(f"Saved to {output_dir}")