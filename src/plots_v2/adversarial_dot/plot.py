from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_EPSILON_LABELS = {
    round(1 / 255, 6): "1/255",
    round(2 / 255, 6): "2/255",
    round(4 / 255, 6): "4/255",
    round(8 / 255, 6): "8/255",
}

_EPSILON_ALPHAS = {
    round(1 / 255, 6): 0.4,
    round(2 / 255, 6): 0.6,
    round(4 / 255, 6): 0.8,
    round(8 / 255, 6): 1.0,
}

_COLOR_FRAGILE = "#C0392B"
_COLOR_NON_FRAGILE = "#2980B9"


def load_adversarial_results(directory: str) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in Path(directory).glob("*.csv")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_adversarial_dot_plot(
    df: pd.DataFrame,
    attack: str,
    output_path: str,
    epsilon_labels: dict[float, str] | None = None,
) -> None:
    if epsilon_labels is None:
        epsilon_labels = _EPSILON_LABELS

    df = df.copy()
    df["_eps_r"] = df["epsilon"].round(6)

    sub = df[df["attack"] == attack]
    if sub.empty:
        return

    epsilons = sorted(sub["_eps_r"].unique())
    n_eps = len(epsilons)

    is_fragile = bool(sub["is_fragile"].iloc[0])
    color = _COLOR_FRAGILE if is_fragile else _COLOR_NON_FRAGILE

    fig, ax = plt.subplots(figsize=(14, max(3, n_eps * 0.9) + 1.0), dpi=150)
    fig.patch.set_facecolor("white")

    for j, eps in enumerate(epsilons):
        row_df = sub[sub["_eps_r"] == eps]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        alpha = _EPSILON_ALPHAS.get(round(eps, 6), 0.8)
        ax.plot(
            [row.baseline_acc, row.adv_acc],
            [j, j],
            color=color,
            alpha=alpha,
            linewidth=1.5,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            row.baseline_acc, j, "o", color=color, alpha=alpha, markersize=6, zorder=3
        )
        ax.plot(
            row.adv_acc,
            j,
            "o",
            color=color,
            alpha=alpha,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )

    ax.set_xlim(0, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0.0, 1.1, 0.1)], fontsize=12)
    ax.set_ylim(n_eps - 0.5, -0.5)

    if attack == "fgsm":
        tick_labels = [epsilon_labels.get(round(e, 6), f"{e:.5f}") for e in epsilons]
        ax.set_yticks(range(n_eps))
        ax.set_yticklabels(
            [f"ε={lbl}" for lbl in tick_labels], fontsize=12, family="monospace"
        )
    else:
        ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.set_xlabel("Accuracy", fontsize=14)
    ax.set_title(attack.upper(), fontsize=14, pad=6)
    ax.tick_params(left=False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adversarial_multi_class(
    df: pd.DataFrame,
    attack: str,
    output_path: str,
    epsilon_labels: dict[float, str] | None = None,
) -> None:
    if epsilon_labels is None:
        epsilon_labels = _EPSILON_LABELS

    df = df.copy()
    df["_eps_r"] = df["epsilon"].round(6)

    sub_attack = df[df["attack"] == attack]
    if sub_attack.empty:
        return

    synsets = sorted(df["synset"].unique())
    n_classes = len(synsets)

    epsilons = sorted(df["_eps_r"].unique())
    M = len(epsilons)
    group_gap = 1.5
    step = M + group_gap

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 3), 10), dpi=150)
    fig.patch.set_facecolor("white")

    bar_positions: list[float] = []
    bar_eps_labels: list[str] = []

    for i, synset in enumerate(synsets):
        sub_class = sub_attack[sub_attack["synset"] == synset]

        if not sub_class.empty:
            class_name = sub_class["class_name"].iloc[0].replace("_", " ")
            is_fragile = bool(sub_class["is_fragile"].iloc[0])
        else:
            class_name = synset
            is_fragile = False
        color = _COLOR_FRAGILE if is_fragile else _COLOR_NON_FRAGILE

        group_center = i * step + (M - 1) / 2

        for j, eps in enumerate(epsilons):
            pos = i * step + j
            bar_positions.append(pos)
            bar_eps_labels.append(
                epsilon_labels.get(round(eps, 6), f"{eps:.5f}")
            )

            row_df = sub_class[sub_class["_eps_r"] == eps]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            alpha = _EPSILON_ALPHAS.get(round(eps, 6), 0.8)

            ax.vlines(
                x=pos,
                ymin=row.adv_acc,
                ymax=row.baseline_acc,
                color=color,
                alpha=alpha,
                linewidth=2,
                zorder=2,
            )
            ax.scatter(pos, row.baseline_acc, s=40, color=color, alpha=alpha, zorder=3)
            ax.scatter(
                pos, row.adv_acc,
                s=40, facecolors="white", edgecolors=color,
                linewidths=1.5, alpha=alpha, zorder=3,
            )

        ax.text(
            group_center,
            -0.18,
            f"{class_name}\n{synset}",
            ha="center",
            va="top",
            fontsize=22,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(
        [f"ε={lbl}" for lbl in bar_eps_labels],
        fontsize=20,
        rotation=45,
        ha="right",
    )

    total_width = (n_classes - 1) * step + M
    ax.set_xlim(-0.8, total_width - 1 + 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.1)], fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.set_title(attack.upper(), fontsize=24, pad=6)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.tick_params(bottom=True, length=4)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
