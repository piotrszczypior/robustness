from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


_COLOR_FRAGILE = "#C0392B"
_COLOR_NON_FRAGILE = "#2980B9"

_SEVERITY_ALPHAS = {1: 0.4, 2: 0.6, 3: 0.8, 4: 0.9, 5: 1.0}


def plot_fragile_dot(
    clean_acc: dict[str, float],
    severity_acc: dict[int, dict[str, float]],
    robust_synsets: list[str],
    fragile_synsets: list[str],
    title: str,
    output_path: str | Path,
    label_map: dict[str, str] | None = None,
) -> None:
    if label_map is None:
        label_map = {}

    severities = sorted(severity_acc)
    M = len(severities)
    group_gap = 1.5
    step = M + group_gap

    all_synsets = list(robust_synsets) + list(fragile_synsets)
    n_classes = len(all_synsets)

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 3), 10), dpi=150)
    fig.patch.set_facecolor("white")

    bar_positions: list[float] = []
    bar_sev_labels: list[str] = []

    for i, synset in enumerate(all_synsets):
        is_fragile = synset in fragile_synsets
        color = _COLOR_FRAGILE if is_fragile else _COLOR_NON_FRAGILE

        group_center = i * step + (M - 1) / 2
        clean = clean_acc.get(synset, float("nan"))

        for j, sev in enumerate(severities):
            pos = i * step + j
            bar_positions.append(pos)
            bar_sev_labels.append(f"S{sev}")

            corrupt = severity_acc[sev].get(synset, float("nan"))
            alpha = _SEVERITY_ALPHAS.get(sev, 1.0)

            if not (np.isnan(clean) or np.isnan(corrupt)):
                ax.vlines(
                    x=pos,
                    ymin=min(clean, corrupt),
                    ymax=max(clean, corrupt),
                    color=color,
                    alpha=alpha,
                    linewidth=2,
                    zorder=2,
                )
            if not np.isnan(clean):
                ax.scatter(pos, clean, s=40, color=color, alpha=alpha, zorder=3)
            if not np.isnan(corrupt):
                ax.scatter(
                    pos, corrupt,
                    s=40, facecolors="white", edgecolors=color,
                    linewidths=1.5, alpha=alpha, zorder=3,
                )

        class_name = label_map.get(synset, synset).replace("_", " ")
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
    ax.set_xticklabels(bar_sev_labels, fontsize=20, rotation=0, ha="center")

    total_width = (n_classes - 1) * step + M
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
