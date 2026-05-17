"""
Scatter: clean accuracy vs mCE per class, one panel per model.
Highlights non-robust classes and shows regression trend.
"""

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

from mce import load_and_aggregate_results, aggregate_for_rmce, compute_rmce_mce

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["alexnet", "resnet18", "resnet50", "vgg16", "vgg19"]
DATA_DIR = "results"

NONROBUST_THRESHOLD = 1.5  # mCE above this → labelled
N_LABELS_MAX = 10  # max labels per panel to avoid clutter

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("tab10", len(MODELS))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all(models: list[str], data_dir: str) -> pd.DataFrame:
    df_alexnet = load_and_aggregate_results("alexnet", data_dir)
    agg_alex = aggregate_for_rmce(df_alexnet, "all")

    rows = []
    for model_name in models:
        df_model = load_and_aggregate_results(model_name, data_dir)
        agg_model = aggregate_for_rmce(df_model, "all")
        metrics = compute_rmce_mce(agg_model, agg_alex, "all")

        clean_acc = agg_model.set_index("synset")["clean"]
        metrics["clean_acc"] = metrics["synset"].map(clean_acc)
        metrics["model"] = model_name
        rows.append(metrics)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Scatter panel
# ---------------------------------------------------------------------------


def _panel(ax: plt.Axes, df: pd.DataFrame, model: str, color, synset_to_name: dict):
    sub = df[df["model"] == model].copy()

    # background points
    ax.scatter(
        sub["clean_acc"],
        sub["mCE"],
        color=color,
        alpha=0.25,
        s=14,
        linewidths=0,
        zorder=2,
    )

    # regression line
    sns.regplot(
        data=sub,
        x="clean_acc",
        y="mCE",
        ax=ax,
        color=color,
        scatter=False,
        line_kws={"linewidth": 2.0, "zorder": 3},
        ci=90,
    )

    # AlexNet reference
    ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    # non-robust outliers
    nonrobust = sub[sub["mCE"] > NONROBUST_THRESHOLD].nlargest(N_LABELS_MAX, "mCE")
    if not nonrobust.empty:
        ax.scatter(
            nonrobust["clean_acc"],
            nonrobust["mCE"],
            color=color,
            edgecolors="white",
            linewidths=0.6,
            s=40,
            zorder=4,
        )
        for _, row in nonrobust.iterrows():
            name = synset_to_name.get(row["synset"], row["synset"])
            ax.annotate(
                name,
                xy=(row["clean_acc"], row["mCE"]),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=6.5,
                color="#222222",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=5,
            )

    ax.set_title(model, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Clean accuracy per class", fontsize=9)
    ax.set_ylabel("mCE", fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(models: list[str], data_dir: str, output: str, synset_map: str | None):
    print("Loading data...")
    df = load_all(models, data_dir)

    # optional synset → human name mapping
    synset_to_name: dict[str, str] = {}
    if synset_map:
        import json

        with open(synset_map) as f:
            idx = json.load(f)
        synset_to_name = {v[0]: v[1].replace("_", " ") for v in idx.values()}

    non_alex = [m for m in models if m != "alexnet"]
    ncols = 2
    nrows = int(np.ceil(len(non_alex) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 7, nrows * 5),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    # shared axis limits
    x_min, x_max = df["clean_acc"].quantile(0.01), df["clean_acc"].quantile(0.99)
    y_min, y_max = 0.0, df["mCE"].clip(upper=3.0).quantile(0.995)

    palette = sns.color_palette("tab10", len(non_alex))

    for ax, model, color in zip(axes, non_alex, palette):
        _panel(ax, df, model, color, synset_to_name)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # hide unused panels
    for ax in axes[len(non_alex) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Clean accuracy vs mCE per class — non-robust classes highlighted",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default="plots/scatter_mce.pdf")
    parser.add_argument(
        "--synset-map",
        default=None,
        help="Path to imagenet_class_index.json for human-readable labels",
    )
    args = parser.parse_args()

    main(args.models, args.data_dir, args.output, args.synset_map)
