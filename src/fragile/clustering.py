from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan

FEATURES = ["acc_clean", "acc_corrupt", "rel_drop", "abs_drop"]
Projection = Literal["pca", "umap"]


def run_clustering(df: pd.DataFrame, projection: Projection = "umap") -> None:
    df = run_hdbscan(df)
    df = run_projection(df, projection=projection)
    plot_clustering(df, projection=projection)


def run_kmeans(features: pd.DataFrame, k: int = 3, random_state: int = 42) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return kmeans.fit_predict(X_scaled)


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


def plot_kmeans(
    df: pd.DataFrame,
    projection: Projection = "umap",
    output_path: str = "images/fragile/clustering",
    filename: str = None,
    fragile_cluster_id: int | None = None,
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    proj = df.attrs.get("projection", projection).upper()
    ev = df.attrs.get("explained_variance", [0, 0])
    x_label = f"{proj}1 ({ev[0]:.1%})" if projection == "pca" else f"{proj}1"
    y_label = f"{proj}2 ({ev[1]:.1%})" if projection == "pca" else f"{proj}2"

    fig, ax = plt.subplots(figsize=(7, 6))
    clusters = sorted(df["cluster"].dropna().unique())
    
    cmap = plt.get_cmap("Set2")
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    if fragile_cluster_id is not None:
        color_map[fragile_cluster_id] = "crimson"

    for cluster, group in df.groupby("cluster"):
        is_fragile = fragile_cluster_id is not None and cluster == fragile_cluster_id
        
        c = color_map[cluster]
        # if fragile_cluster_id is not None and not is_fragile:
        #     c = "lightgray"

        ax.scatter(
            group["dim1"], group["dim2"],
            c=[c],
            label=f"cluster {int(cluster)}" + (" (fragile)" if is_fragile else ""),
            alpha=0.9 if is_fragile else 0.5,
            s=30 if is_fragile else 12,
            zorder=3 if is_fragile else 1,
        )

    ax.set_title(f"{proj} colored by K-Means cluster (k={len(clusters)})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    filename = f"{projection}_kmeans.png" if filename is None else filename
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_dir}")