from __future__ import annotations
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plots.base import BasePlotPipeline
from plots.data import get_data

logger = logging.getLogger(__name__)


def _calculate_accuracy_per_class(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["y_true"])["is_correct"].mean()


def violin_box_with_markers(ax, data, y_pos, label):
    data = np.asarray(data)

    parts = ax.violinplot(
        [data],
        positions=[y_pos],
        vert=False,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for b in parts["bodies"]:
        b.set_alpha(0.6)
        b.set_facecolor("#6D96D8")

    ax.boxplot(
        [data],
        positions=[y_pos],
        vert=False,
        widths=0.05,
        patch_artist=True,
        boxprops=dict(facecolor="none"),
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    mean = float(data.mean())
    p5 = float(np.percentile(data, 5))
    p10 = float(np.percentile(data, 10))

    ax.vlines(mean, y_pos - 0.12, y_pos + 0.12, linewidth=3)
    ax.vlines(p5, y_pos - 0.12, y_pos + 0.12, linewidth=3)
    ax.vlines(p10, y_pos - 0.12, y_pos + 0.12, linewidth=3)

    std = float(data.std(ddof=0))
    ax.text(
        1.02,
        y_pos,
        f"{label}\n({mean * 100:.2f} ± {std * 100:.2f})",
        va="center",
        transform=ax.get_yaxis_transform(),
    )

    return mean, std, p5, p10


class ViolinPlot(BasePlotPipeline):
    def _setup_canvas(self):
        content = self.config.content
        models = content.models
        # Adjust height based on number of violins (2 per model: clean & corrupted)
        num_violins = len(models) * 2
        figsize = (12, max(4, num_violins * 1.5))
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(self.config.title, pad=20, fontsize=14, fontweight="bold")

    def transform_data(self):
        content = self.config.content
        models = content.models

        plot_data = []
        for model in models:
            clean_df = get_data(self.data_dir, model.clean)
            corrupted_df = get_data(self.data_dir, model.corrupted)

            clean_acc = _calculate_accuracy_per_class(clean_df)
            corrupted_acc = _calculate_accuracy_per_class(corrupted_df)

            plot_data.append(
                {
                    "name": model.name,
                    "clean_acc": clean_acc.values,
                    "corrupted_acc": corrupted_acc.values,
                    "corruption_label": model.corruption_label,
                }
            )

        return plot_data

    def render(self, data: list[dict]):
        y_pos = 1
        y_ticks = []
        y_labels = []

        # We plot from bottom to top
        for entry in reversed(data):
            # Corrupted first (lower position)
            violin_box_with_markers(
                self.ax,
                entry["corrupted_acc"],
                y_pos=y_pos,
                label=f"{entry['name']} ({entry['corruption_label']})",
            )
            y_ticks.append(y_pos)
            y_labels.append(entry["corruption_label"])
            y_pos += 1

            # Clean second (higher position)
            violin_box_with_markers(
                self.ax,
                entry["clean_acc"],
                y_pos=y_pos,
                label=f"{entry['name']} (Clean)",
            )
            y_ticks.append(y_pos)
            y_labels.append("Clean")
            y_pos += 1

            # Add some spacing between models if multiple
            y_pos += 0.5

        self.ax.set_xlim(0, 1)
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(y_labels)
        self.ax.set_xlabel("Per-class accuracy")

        plt.tight_layout()
