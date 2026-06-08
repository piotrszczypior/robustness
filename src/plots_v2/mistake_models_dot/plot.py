from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


_MODEL_COLORS = ["#2980B9", "#C0392B", "#27AE60", "#8E44AD", "#E67E22", "#16A085"]


def render_models(
    model_entries: dict[str, list[dict]],
    synset: str,
    synset_label: str,
    models: list[str],
    model_labels: dict[str, str],
    max_classes: int,
    output_path: Path,
) -> None:
    # Build union of predicted classes, ranked by total count across models
    class_totals: dict[tuple[str, str], int] = defaultdict(int)
    class_is_correct: dict[tuple[str, str], bool] = {}
    for entries in model_entries.values():
        for e in entries:
            key = (e["label"], e["synset"])
            class_totals[key] += e["count"]
            class_is_correct[key] = e["is_correct"]

    sorted_classes = sorted(class_totals, key=lambda k: -class_totals[k])[:max_classes]
    n_classes = len(sorted_classes)

    if n_classes == 0:
        print(f"  No data for {synset}, skipping.")
        return

    # Per-model lookup: (label, pred_synset) -> count
    model_lookup: dict[str, dict[tuple[str, str], int]] = {
        model: {(e["label"], e["synset"]): e["count"] for e in entries}
        for model, entries in model_entries.items()
    }

    M = len(models)
    group_gap = 1.5
    step = M + group_gap
    total_width = (n_classes - 1) * step + M

    y_max = max((v for entries in model_entries.values() for e in entries for v in [e["count"]]), default=10)
    tick_step = max(1, round(y_max / 8))
    yticks = list(range(0, y_max + tick_step + 1, tick_step))

    fig, ax = plt.subplots(
        figsize=(max(14, n_classes * M * 1.2 + 4), 8),
        dpi=300,
    )
    fig.patch.set_facecolor("white")

    bar_positions: list[float] = []
    bar_xlabels: list[str] = []

    for i, (pred_label, pred_synset) in enumerate(sorted_classes):
        group_center = i * step + (M - 1) / 2
        is_correct = class_is_correct.get((pred_label, pred_synset), False)

        for j, model in enumerate(models):
            pos = i * step + j
            bar_positions.append(pos)
            bar_xlabels.append(model_labels.get(model, model))

            color = _MODEL_COLORS[j % len(_MODEL_COLORS)]
            count = model_lookup[model].get((pred_label, pred_synset), 0)

            if count > 0:
                ax.vlines(pos, 0, count, color=color, linewidth=2, zorder=2)
            if is_correct:
                ax.scatter(pos, count, s=40, color=color, zorder=3)
            else:
                ax.scatter(
                    pos, count,
                    s=40, facecolors="white", edgecolors=color,
                    linewidths=1.5, zorder=3,
                )

        ax.text(
            group_center,
            -0.06,
            f"{pred_label}\n({pred_synset})",
            ha="center",
            va="top",
            fontsize=12,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_xlabels, fontsize=11, ha="center")
    ax.set_xlim(-0.8, total_width - 1 + 0.8)
    ax.set_ylim(0, y_max * 1.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#eeeeee", linewidth=1, zorder=0)
    ax.set_ylabel("Prediction Count", fontsize=14)
    ax.set_title(synset_label, fontsize=18, pad=6)
    ax.tick_params(bottom=True, length=4)

    legend_handles = [
        mpatches.Patch(facecolor=_MODEL_COLORS[j % len(_MODEL_COLORS)], label=model_labels.get(m, m))
        for j, m in enumerate(models)
    ]
    ax.legend(handles=legend_handles, fontsize=12, framealpha=0.8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
