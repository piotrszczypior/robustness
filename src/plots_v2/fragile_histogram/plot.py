from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_BLUE = "#4D99CB"
_RED = "#C54C3F"

def render(df: pd.DataFrame, output_path: Path, title: str = "") -> None:
    df = df[df["acc_clean"] > 0].copy()
    rel_drop = (df["acc_clean"] - df["acc_corrupt"]) / df["acc_clean"]
    rel_drop = rel_drop.dropna().values

    n_negative = int((rel_drop < 0).sum())
    rel_drop_clipped = rel_drop[rel_drop >= 0]

    p75 = np.percentile(rel_drop_clipped, 75)

    counts, bin_edges = np.histogram(rel_drop_clipped, bins=40)
    colors = [_RED if left >= p75 else _BLUE for left in bin_edges[:-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        color=colors,
        align="edge",
        edgecolor="none",
    )
    ax.axvline(p75, color=_RED, linestyle="--", linewidth=1)

    # if n_negative > 0:
    #     ax.text(
    #         -0.1,
    #         counts.max() * 0.97,
    #         f"neg: {n_negative}",
    #         fontsize=9,
    #         color="gray",
    #         ha="left",
    #         va="top",
    #     )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Relative accuracy drop", fontsize=11)
    ax.set_ylabel("Class count", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([-0.1, bin_edges[-1]])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
