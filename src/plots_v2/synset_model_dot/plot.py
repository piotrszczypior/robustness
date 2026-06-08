from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_MODEL_COLORS = [
    "#2980B9",
    "#C0392B",
    "#27AE60",
    "#8E44AD",
    "#E67E22",
    "#16A085",
]


def plot_synset_model_dot(
    clean_acc: dict[str, dict[str, float]],
    corrupt_acc: dict[str, dict[str, float]],
    synsets: list[str],
    models: list[str],
    title: str,
    output_path: str | Path,
    label_map: dict[str, str] | None = None,
    model_labels: dict[str, str] | None = None,
) -> None:
    """Cleveland dot plot — clean vs corrupt accuracy per synset × model.

    Parameters
    ----------
    clean_acc:    {synset -> {model -> accuracy}}
    corrupt_acc:  {synset -> {model -> accuracy}}
    synsets:      Ordered list of synset IDs (defines x-axis groups).
    models:       Ordered list of model keys (defines ticks within each group).
    title:        Plot title (e.g. "Noise\\nseverity 3").
    label_map:    {synset -> human label}
    model_labels: {model_key -> display name}
    """
    if label_map is None:
        label_map = {}
    if model_labels is None:
        model_labels = {}

    M = len(models)
    group_gap = 1.5
    step = M + group_gap
    n_synsets = len(synsets)

    fig, ax = plt.subplots(
        figsize=(max(12, n_synsets * M * 1.8), 8), dpi=150
    )
    fig.patch.set_facecolor("white")

    bar_positions: list[float] = []
    bar_labels: list[str] = []

    for i, synset in enumerate(synsets):
        group_center = i * step + (M - 1) / 2

        for j, model in enumerate(models):
            pos = i * step + j
            bar_positions.append(pos)
            bar_labels.append(model_labels.get(model, model))

            color = _MODEL_COLORS[j % len(_MODEL_COLORS)]
            clean = clean_acc.get(synset, {}).get(model, float("nan"))
            corrupt = corrupt_acc.get(synset, {}).get(model, float("nan"))

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

        class_name = label_map.get(synset, synset).replace("_", " ")
        ax.text(
            group_center,
            -0.06,
            f"{class_name}\n{synset}",
            ha="center",
            va="top",
            fontsize=22,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=13, ha="center")

    total_width = (n_synsets - 1) * step + M
    ax.set_xlim(-0.8, total_width - 1 + 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.0, 1.1, 0.1)], fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.set_title(title, fontsize=24, pad=6)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.tick_params(bottom=True, length=4)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
