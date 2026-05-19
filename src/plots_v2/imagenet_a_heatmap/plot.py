from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_synset_to_imagenet_a_index, get_synset_to_imagenet_r_index
from dataset import DATASET_ALIAS_TO_LABEL

_SYNSET_TO_INDEX = {
    "imagenet_a": get_synset_to_imagenet_a_index,
    "imagenet_r": get_synset_to_imagenet_r_index,
}


def build_matrix(
    dfs: dict[str, pd.DataFrame], dataset: str, sort_by: str | None = None
) -> pd.DataFrame:
    if sort_by and sort_by in dfs:
        class_order = (
            dfs[sort_by]
            .set_index("synset")["acc_corrupt"]
            .sort_values(ascending=False)
            .index
        )
    else:
        class_order = (
            pd.concat(list(dfs.values()))
            .groupby("synset")["acc_corrupt"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

    wide = pd.DataFrame(
        {name: df.set_index("synset")["acc_corrupt"] for name, df in dfs.items()}
    ).T
    wide = wide.reindex(class_order, axis=1)

    synset_to_index = _SYNSET_TO_INDEX[dataset]()
    wide.columns = [synset_to_index[s] for s in class_order]

    return wide


_THRESHOLDS = [
    (0.7, "#1A5276", "mean < 0.7"),
    (0.5, "#7D6608", "mean < 0.5"),
    (0.3, "#7B241C", "mean < 0.3"),
]


def _draw_thresholds(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    mean_sorted = matrix.mean(axis=0)  # already sorted descending
    tick_positions = []
    tick_labels = []
    tick_colors = []

    for threshold, color, label in _THRESHOLDS:
        idx = int((mean_sorted > threshold).sum())
        if 0 < idx < len(mean_sorted):
            ax.axvline(idx, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
            tick_positions.append(idx)
            tick_labels.append(label)
            tick_colors.append(color)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)


def render(matrix: pd.DataFrame, out_path: Path, dataset: str = "imagenet_a") -> None:
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(20, max(8, len(matrix) * 0.6)))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar=True,
        cbar_kws={"label": f"Accuracy ({DATASET_ALIAS_TO_LABEL[dataset]})"},
        ax=ax,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        alpha=0.7,
    )

    _draw_thresholds(ax, matrix)

    for y in range(1, len(matrix)):
        ax.axhline(y, color="white", linewidth=0.8)

    plt.setp(
        ax.get_yticklabels(),
        fontfamily="monospace",
        fontsize=12,
        rotation=0,
        color="#222222",
    )

    ax.set_xlabel(
        f"{DATASET_ALIAS_TO_LABEL[dataset]} synsets ordered by cross-model mean accuracy",
        fontsize=13,
        color="#444444",
        labelpad=6,
    )
    ax.set_ylabel("")
    ax.tick_params(left=False)
    ax.spines[:].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="x", length=4, width=0.8, color="#aaaaaa", bottom=True)
    ax.set_ylim(len(matrix) + 0.1, -0.1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
