from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def build_rmce_matrix(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wide = pd.DataFrame(
        {name: df.set_index("y_true")["RmCE"] for name, df in dfs.items()}
    ).T
    return wide.reindex(sorted(wide.columns), axis=1)


def render(matrix: pd.DataFrame, out_path: Path) -> None:
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(20, max(8, len(matrix) * 0.6)))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=1.0,
        vmin=0,
        vmax=3,
        cbar=True,
        cbar_kws={"label": "RmCE"},
        ax=ax,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
    )

    for y in range(1, len(matrix)):
        ax.axhline(y, color="white", linewidth=0.8)

    plt.setp(
        ax.get_yticklabels(),
        fontfamily="monospace",
        fontsize=12,
        rotation=0,
        color="black",
    )

    n_classes = matrix.shape[1]
    tick_positions = list(range(0, n_classes, 25)) + [n_classes - 1]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(
        [str(p) for p in tick_positions],
        fontsize=11,
        color="black",
        rotation=0,
    )

    ax.set_xlabel("ImageNet class index", fontsize=13, color="black", labelpad=6)
    ax.set_ylabel("")
    ax.tick_params(left=False)
    ax.spines[:].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.tick_params(axis="x", length=4, width=0.8, color="#aaaaaa", bottom=True)
    ax.set_ylim(len(matrix) + 0.1, -0.1)

    plt.tight_layout()
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, out_path)
    plt.close(fig)
