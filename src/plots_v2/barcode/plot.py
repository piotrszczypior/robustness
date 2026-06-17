from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


COMMON_THRESHOLD = 15


def build_barcode_matrix(
    flagged_dfs: dict[str, pd.DataFrame], flag_col: str
) -> pd.DataFrame:
    wide = (
        pd.DataFrame(
            {name: df.set_index("y_true")[flag_col] for name, df in flagged_dfs.items()}
        )
        .fillna(0)
        .astype(int)
        .T
    )

    fragile_counts = wide.sum(axis=0)
    common = set(fragile_counts[fragile_counts >= COMMON_THRESHOLD].index)

    result = wide.copy()
    for cls in common:
        result[cls] = result[cls].apply(lambda v: 2 if v == 1 else v)

    return result.sort_index(axis=1)


def render(matrix: pd.DataFrame, out_path: Path, y_label = True) -> None:
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(20, max(8, len(matrix) * 0.6)))
    fig.patch.set_facecolor("white")

    cmap = ListedColormap(["#F0F0F0", "#1f77b4", "red"])

    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=2,
        cbar=False,
        ax=ax,
        xticklabels=False,
        yticklabels=y_label,
        linewidths=0,
    )

    for y in range(1, len(matrix)):
        ax.axhline(y, color="white", linewidth=0.8)

    plt.setp(
        ax.get_yticklabels(),
        fontfamily="monospace",
        fontsize=14,
        rotation=0,
        color="#222222",
    )

    n_classes = matrix.shape[1]
    tick_step = 25
    tick_positions = list(range(0, n_classes, tick_step)) + [n_classes - 1]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(
        [str(p) for p in tick_positions],
        fontsize=13,
        color="black",
        rotation=0,
    )

    ax.set_xlabel("ImageNet classes", fontsize=18, color="black", labelpad=6)
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
