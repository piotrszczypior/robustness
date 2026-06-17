from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render(df: pd.DataFrame, output_path: Path, title: str = "") -> None:
    df = df.sort_values("acc_clean", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))

    fragile_mask = (df["acc_clean"] >= 0.8) & (df["acc_corrupt"] <= 0.5)
    normal_mask = ~fragile_mask

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Class index (sorted by clean accuracy)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    n = len(df)
    x_ticks = list(range(0, n, max(1, n // 10)))
    if (n - 1) not in x_ticks:
        x_ticks.append(n - 1)
    ax.set_xticks(x_ticks)
    ax.set_xlim([0, n - 1])
    ax.set_ylim(0, 1.05)

    ax.plot(x, df["acc_clean"].values, color="black", linewidth=2, label="Clean")
    ax.scatter(
        x[normal_mask],
        df.loc[normal_mask, "acc_corrupt"].values,
        color="#1f77b4",
        alpha=0.6,
        s=18,
        label="Corrupted",
        zorder=3,
        edgecolors="none",
    )
    ax.scatter(
        x[fragile_mask],
        df.loc[fragile_mask, "acc_corrupt"].values,
        color="red",
        alpha=0.6,
        s=18,
        label="Fragile (clean≥0.8, corr≤0.5)",
        zorder=4,
        edgecolors="none",
    )

    # ax.legend(bbox_to_anchor=(1.01, 1.01), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, output_path)
    plt.close(fig)
