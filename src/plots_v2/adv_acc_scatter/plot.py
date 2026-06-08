from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats
from model import MODELS


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.tick_params(left=True, labelsize=11)


def plot_scatter(
    df: pd.DataFrame,
    model: str,
    attack: str,
    epsilon: int,
    corruption_label: str,
    output_dir: Path,
    top_n: int = 3,
) -> None:
    """Scatter: ImageNet-C accuracy (x) vs adversarial accuracy (y), one point per class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    fragile = df[df["is_strongly_fragile"] == 1]
    non_frgile = df[df["is_strongly_fragile"] == 0]
    print(df.head())

    ax.scatter(fragile["acc_clean"], fragile["adv_acc"], s=25, alpha=0.6, color="red", zorder=4)
    ax.scatter(non_frgile["acc_clean"], non_frgile["adv_acc"], s=18, alpha=0.6, color="#2980B9", zorder=3)


    bottom = fragile.nsmallest(top_n, "adv_acc")
    top = fragile.nlargest(top_n, "adv_acc")

    labels = pd.concat([bottom, top]).sort_values("adv_acc", ascending=False).reset_index(drop=True)
    n = len(labels)
    label_xs = np.linspace(0.8, 0.3, n)
    label_ys = np.linspace(0.8, 0.3, n)
    shift = [0.2, 0.3, 0.35, 0, 0, 0,]
    for (i, row), ly, lx in zip(labels.iterrows(), label_ys, label_xs):
        ax.annotate(
            row["class_name"],
            xy=(row["acc_clean"], row["adv_acc"]),
            xytext=(lx, ly + shift[i]),
            fontsize=7,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        )

    # rho, pval = stats.spearmanr(df["corrupt_acc"], df["adv_acc"])
    # ax.text(
    #     0.97, 0.97,
    #     f"ρ = {rho:.2f}\np = {pval:.3f}",
    #     transform=ax.transAxes, ha="right", va="top", fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    # )

    eps_str = f"{epsilon}/255"
    ax.set_xlabel(f"ImageNet accuracy", fontsize=12)
    ax.set_ylabel(f"Adversarial accuracy  ({attack.upper()} ε={eps_str})", fontsize=12)
    ax.set_title(f"{MODELS[model]}  ·  {attack.upper()} ε={eps_str}  ·  {corruption_label}", fontsize=13, pad=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))


    _style(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_{attack}_{epsilon}_255_{corruption_label.lower().replace(" ", "_")}"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)



def plot_scatter_robust(
    df: pd.DataFrame,
    model: str,
    attack: str,
    epsilon: int,
    corruption_label: str,
    output_dir: Path,
    top_n: int = 2,
) -> None:
    """Scatter: ImageNet-C accuracy (x) vs adversarial accuracy (y), one point per class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    fragile = df[df["robust"] == 1]
    non_frgile = df[df["robust"] == 0]
    print(df.head())

    ax.scatter(fragile["acc_clean"], fragile["adv_acc"], s=25, alpha=0.6, color="#07A93D", zorder=4)
    ax.scatter(non_frgile["acc_clean"], non_frgile["adv_acc"], s=18, alpha=0.6, color="#2980B9", zorder=3)

    bottom = fragile.nsmallest(top_n, "adv_acc")
    top = fragile.nlargest(top_n, "adv_acc")

    labels = pd.concat([bottom, top]).sort_values("adv_acc", ascending=False).reset_index(drop=True)
    n = len(labels)
    label_xs = np.linspace(0.8, 0.3, n)
    label_ys = np.linspace(0.8, 0.3, n)
    shift = [0.2, 0.3, 0.35, 0, 0, 0,]
    for (i, row), ly, lx in zip(labels.iterrows(), label_ys, label_xs):
        ax.annotate(
            row["class_name"],
            xy=(row["acc_clean"], row["adv_acc"]),
            xytext=(lx, ly + shift[i]),
            fontsize=7,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        )

    # rho, pval = stats.spearmanr(df["corrupt_acc"], df["adv_acc"])
    # ax.text(
    #     0.97, 0.97,
    #     f"ρ = {rho:.2f}\np = {pval:.3f}",
    #     transform=ax.transAxes, ha="right", va="top", fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    # )

    eps_str = f"{epsilon}/255"
    ax.set_xlabel(f"ImageNet accuracy", fontsize=12)
    ax.set_ylabel(f"Adversarial accuracy  ({attack.upper()} ε={eps_str})", fontsize=12)
    ax.set_title(f"{MODELS[model]}  ·  {attack.upper()} ε={eps_str}  ·  {corruption_label}", fontsize=13, pad=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))


    _style(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_robust_{attack}_{epsilon}_255_{corruption_label.lower().replace(" ", "_")}"
    for ext in ("png", "pdf"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


