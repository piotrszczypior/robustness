from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_COMBO_COLORS = ["#2980B9", "#C0392B", "#27AE60", "#8E44AD", "#E67E22", "#16A085"]


def plot_model_dot(
    clean_acc: dict[str, dict[str, float]],
    combo_acc: dict[tuple[str, int], dict[str, float]],
    models: list[str],
    synset: str,
    combos: list[tuple[str, int]],
    title: str,
    output_path: str | Path,
    model_labels: dict[str, str] | None = None,
) -> None:
    if model_labels is None:
        model_labels = {}

    M = len(models)
    n_combos = len(combos)
    group_gap = 1.5
    step = M + group_gap
    total_width = (n_combos - 1) * step + M

    fig, ax = plt.subplots(
        figsize=(max(12, n_combos * M * 1.5 + 4), 8),
        dpi=300,
    )
    fig.patch.set_facecolor("white")

    bar_positions: list[float] = []
    bar_mlabels: list[str] = []

    for i, (corruption, severity) in enumerate(combos):
        color = _COMBO_COLORS[i % len(_COMBO_COLORS)]
        group_center = i * step + (M - 1) / 2

        for j, model in enumerate(models):
            pos = i * step + j
            bar_positions.append(pos)
            bar_mlabels.append(model_labels.get(model, model))

            clean = clean_acc.get(model, {}).get(synset, float("nan"))
            corrupt = combo_acc.get((corruption, severity), {}).get(model, float("nan"))

            if not (np.isnan(clean) or np.isnan(corrupt)):
                ax.vlines(
                    x=pos,
                    ymin=min(clean, corrupt),
                    ymax=max(clean, corrupt),
                    color=color,
                    linewidth=2,
                    zorder=2,
                )
            if not np.isnan(clean):
                ax.scatter(pos, clean, s=40, color=color, zorder=3)
            if not np.isnan(corrupt):
                ax.scatter(
                    pos, corrupt,
                    s=40, facecolors="white", edgecolors=color,
                    linewidths=1.5, zorder=3,
                )

        corruption_display = corruption.capitalize().replace("_", " ")
        ax.text(
            group_center,
            -0.1,
            f"{corruption_display}\nseverity {severity}",
            ha="center",
            va="top",
            fontsize=16,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_mlabels, fontsize=14, ha="center")
    ax.set_xlim(-0.8, total_width - 1 + 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.1)], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_title(title, fontsize=18, pad=6)
    ax.tick_params(bottom=True, length=4)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(out, dpi=300, bbox_inches="tight")
    from utils import save_as_pdf
    save_as_pdf(fig, output_path)
    plt.close(fig)
