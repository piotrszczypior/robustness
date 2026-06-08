from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.tick_params(left=True, labelsize=11)


def plot_scatter(
    df: pd.DataFrame,
    model: str,
    attack: str,
    epsilon: float,
    corruption_label: str,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Scatter: ImageNet-C accuracy (x) vs adversarial accuracy (y), one point per class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    ax.scatter(df["corrupt_acc"], df["adv_acc"], s=30, alpha=0.6, color="#2980B9", zorder=3)

    bottom = df.nsmallest(top_n, "adv_acc")
    texts = []
    for _, row in bottom.iterrows():
        t = ax.text(row["corrupt_acc"], row["adv_acc"], row["class_name"], fontsize=7)
        texts.append(t)

    adjust_text(
        texts,
        x=df["corrupt_acc"].values,
        y=df["adv_acc"].values,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        expand=(2.0, 2.5),
        force_text=(0.8, 1.5),
        iter_lim=500,
    )

    rho, pval = stats.spearmanr(df["corrupt_acc"], df["adv_acc"])
    ax.text(
        0.97, 0.97,
        f"ρ = {rho:.2f}\np = {pval:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    eps_str = f"{round(epsilon * 255)}/255"
    ax.set_xlabel(f"ImageNet-C accuracy  ({corruption_label})", fontsize=13)
    ax.set_ylabel(f"Adversarial accuracy  ({attack.upper()} ε={eps_str})", fontsize=13)
    ax.set_title(f"{model}  ·  {attack.upper()} ε={eps_str}  ·  {corruption_label}", fontsize=12, pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _style(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{attack}_{round(epsilon * 255)}_255_{corruption_label}"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
