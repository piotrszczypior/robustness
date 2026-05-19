from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _violin_box_with_markers(ax, data: np.ndarray, y_pos: float, label: str) -> None:
    parts = ax.violinplot(
        [data],
        positions=[y_pos],
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.6)
        b.set_facecolor("#6D96D8")

    ax.boxplot(
        [data],
        positions=[y_pos],
        vert=False,
        widths=0.05,
        patch_artist=True,
        boxprops=dict(facecolor="none"),
        medianprops=None,
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    mean = float(data.mean())
    p10 = float(np.percentile(data, 10))
    p50 = float(np.percentile(data, 50))
    p90 = float(np.percentile(data, 90))
    std = float(data.std(ddof=0))

    ax.vlines(p10, y_pos - 0.08, y_pos + 0.08, linewidth=2, color="#E8A0A0")
    ax.vlines(p50, y_pos - 0.08, y_pos + 0.08, linewidth=2, color="#DA5445")
    ax.vlines(p90, y_pos - 0.08, y_pos + 0.08, linewidth=2, color="#7B1A11")

    ax.text(
        1.02,
        y_pos,
        f"{label}\n({mean * 100:.2f} ± {std * 100:.2f})",
        va="center",
        transform=ax.get_yaxis_transform(),
    )


def render(dfs: dict[str, pd.DataFrame], out_path: Path, title: str = "") -> None:
    n_models = len(dfs)
    fig, ax = plt.subplots(figsize=(12, max(4, n_models * 3)))
    fig.patch.set_facecolor("white")

    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("Per-class accuracy")

    y_pos = 1
    y_ticks = []
    y_labels = []

    for model_label, df in reversed(list(dfs.items())):
        _violin_box_with_markers(
            ax,
            df["acc_corrupt"].values,
            y_pos=y_pos,
            label=f"{model_label} (Corrupted)",
        )
        y_ticks.append(y_pos)
        y_pos += 1

        _violin_box_with_markers(
            ax,
            df["acc_clean"].values,
            y_pos=y_pos,
            label=f"{model_label} (Clean)",
        )
        y_ticks.append(y_pos)
        y_pos += 1.5

    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
