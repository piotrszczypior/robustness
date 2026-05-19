from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot(df, output_path: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_title(title)
    ax.set_xlabel("Clean accuracy")
    ax.set_ylabel("Corrupted accuracy")

    ax.grid(True, linestyle=":", alpha=0.6)

    ticks = np.arange(0.0, 1.1, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.scatter(
        df["acc_clean"],
        df["acc_corrupt"],
        color="#1f77b4",
        alpha=0.6,
        zorder=3,
        edgecolors="none",
    )
    ax.plot([0, 1.05], [0, 1.05], color="red", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
