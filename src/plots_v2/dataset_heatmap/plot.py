from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    get_synset_to_imagenet_a_index,
    get_synset_to_imagenet_r_index,
    get_synset_to_index_imagenet1k,
)
from dataset import DATASET_ALIAS_TO_LABEL

_SYNSET_TO_INDEX = {
    "imagenet_a": get_synset_to_imagenet_a_index,
    "imagenet_r": get_synset_to_imagenet_r_index,
    "imagenet_c": get_synset_to_index_imagenet1k,
    "imagenet": get_synset_to_index_imagenet1k,
}


def get_dataset(alias):
    relevant = alias.split("_")[:2]
    return "_".join(relevant)

def build_matrix(
    dfs: dict[str, pd.DataFrame], dataset: str, sort_by: str | None = None
) -> pd.DataFrame:
    wide = pd.DataFrame(
        {name: df.set_index("synset")["acc_corrupt"] for name, df in dfs.items()}
    ).T

    synset_to_index = _SYNSET_TO_INDEX[get_dataset(dataset)]()

    if sort_by is None:
        class_order = sorted(wide.columns, key=lambda s: synset_to_index.get(s, 9999))
    elif sort_by == "mean" or sort_by not in wide.index:
        class_order = wide.mean(axis=0).sort_values(ascending=False).index.tolist()
    else:
        class_order = wide.loc[sort_by].sort_values(ascending=False).index.tolist()

    wide = wide[class_order]
    wide.columns = [synset_to_index[s] for s in class_order]

    return wide


def render(
    matrix: pd.DataFrame,
    out_path: Path,
    dataset: str = "imagenet_a",
    sort_by: str | None = None,
) -> None:
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
        cbar_kws={"label": f"Accuracy ({DATASET_ALIAS_TO_LABEL[get_dataset(dataset)]})"},
        ax=ax,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        alpha=0.7,
    )

    ax.set_xticks([])

    for y in range(1, len(matrix)):
        ax.axhline(y, color="white", linewidth=0.8)

    plt.setp(
        ax.get_yticklabels(),
        fontfamily="monospace",
        fontsize=12,
        rotation=0,
        color="#222222",
    )

    # if sort_by is None:
    #     xlabel = f"{DATASET_ALIAS_TO_LABEL[dataset]} synsets ordered by class index"
    # elif sort_by == "mean":
    #     xlabel = f"{DATASET_ALIAS_TO_LABEL[dataset]} synsets ordered by cross-model mean accuracy"
    # else:
    #     xlabel = (
    #         f"{DATASET_ALIAS_TO_LABEL[dataset]} synsets ordered by {sort_by} accuracy"
    #     )
    
    n_classes = 1000

    # if dataset in ["imagenet", "imagenet_c"]:
    # else:
        # n_classes = 200

    # ax.set_xlabel(xlabel, fontsize=13, color="#444444", labelpad=6)
    ax.set_ylabel("")
    tick_positions = list(range(0, n_classes, 25)) + [n_classes - 1]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(
        [str(p) for p in tick_positions],
        fontsize=11,
        color="#555555",
        rotation=0,
    )

    ax.tick_params(left=False)
    ax.spines[:].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="x", length=4, width=0.8, color="#aaaaaa", bottom=True)
    ax.set_ylim(len(matrix) + 0.1, -0.1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
