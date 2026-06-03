from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_PATH = Path("results/representations/resnet50_class_metrics.parquet")
PRED_DIR = Path("embeddings")
OUT_DIR = Path("results/representations/fragility")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "resnet50"
METRIC = "angular_distance_median"


def clean_pred_path():
    return PRED_DIR / f"{MODEL}_imagenet_embeddings.parquet"


def cond_pred_path(corruption, group, severity):
    return PRED_DIR / f"{MODEL}_imagenet_c_{group}_{corruption}_{severity}_embeddings.parquet"


def per_class_accuracy(path):
    p = pd.read_parquet(path, columns=["synset", "y_true", "y_pred"])
    correct = (p["y_true"] == p["y_pred"]).astype(int)
    return correct.groupby(p["synset"]).mean()


def heatmap(plot_matrix, path):
    data = plot_matrix.values
    n_rows, n_cols = data.shape
    lo, hi = np.nanmin(data), np.nanmax(data)
    
    fig, ax = plt.subplots(figsize=(0.5 * n_cols + 2.5, 0.5 * n_rows + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=lo, vmax=hi)
    
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(plot_matrix.columns, rotation=45, ha="right")
    ax.set_xlabel("corruption")
    
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"s{c}" for c in plot_matrix.index])
    ax.set_ylabel("severity")
    
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if v < 0.67 and v > 0.4 else "white", fontsize=8)
                        
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Spearman rank correlation between per-class cosine distance and accuracy drop")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    df = pd.read_parquet(METRICS_PATH)
    df = df[(df["model"] == MODEL) & (df["metric"] == METRIC)]
    clean_acc = per_class_accuracy(clean_pred_path())

    rows = []
    for (corruption, group, severity), g in df.groupby(["corruption", "group", "severity"]):
        op = cond_pred_path(corruption, group, severity)
        if not op.exists():
            print(f"(missing: {op.name})")
            continue
        rot = g.set_index("synset")["value"]
        drop = (clean_acc - per_class_accuracy(op)).reindex(rot.index)
        rho = spearmanr(rot, drop, nan_policy="omit")[0]
        rows.append({"corruption": corruption, "severity": int(severity), "rho": rho})

    out = pd.DataFrame(rows)
    # Wstępnie pivotujemy tak samo, by łatwo policzyć średnią i posortować
    matrix = out.pivot(index="corruption", columns="severity", values="rho")
    sev_cols = list(matrix.columns)
    matrix["mean"] = matrix[sev_cols].mean(axis=1)
    matrix = matrix.sort_values("mean", ascending=False)

    print(matrix.round(3).to_string())
    matrix.to_csv(OUT_DIR / f"rho_matrix_{MODEL}.csv")
    
    # Do funkcji heatmap podajemy tylko wartości severity (bez kolumny mean) i transponujemy (.T)
    plot_matrix = matrix[sev_cols].T
    heatmap(plot_matrix, OUT_DIR / f"rho_matrix_{MODEL}.png")
    
    print(f"\n-> {OUT_DIR / f'rho_matrix_{MODEL}.csv'}")
    print(f"-> {OUT_DIR / f'rho_matrix_{MODEL}.png'}")


if __name__ == "__main__":
    main()