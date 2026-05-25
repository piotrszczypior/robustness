from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def render(df: pd.DataFrame, model_label: str, out_path: Path) -> None:
    """Scatter plot of RmCE per synset (sorted by y_true index).

    Points with RmCE < 1 are gray; points >= 1 are red.
    A black horizontal line marks y = 1.
    """
    df = df.sort_values("y_true").reset_index(drop=True)
    x = range(len(df))
    rmce = df["RmCE"]

    below = rmce < 1
    above = ~below

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(
        [i for i, m in zip(x, below) if m],
        rmce[below],
        color="#bbbbbb", s=8, zorder=2, label="RmCE < 1",
    )
    ax.scatter(
        [i for i, m in zip(x, above) if m],
        rmce[above],
        color="#c0392b", s=8, zorder=3, label="RmCE ≥ 1",
    )

    ax.axhline(1.0, color="#000000", linewidth=1.0, zorder=1)

    ax.set_xlabel("Class index (sorted)", fontsize=11)
    ax.set_ylabel("RmCE", fontsize=11)
    ax.set_title(model_label, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
