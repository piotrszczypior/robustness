from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt

from plots.base import BasePlotPipeline
from plots.data import get_data, calculate_accuracy_per_class

logger = logging.getLogger(__name__)


class SortedIndexClassDegradation(BasePlotPipeline):
    def schema(self):
        return {
            "baseline": {"label": str, "data": str},
            "degraded": {"label": str, "data": str},
        }

    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(self.config.title)
        self.ax.set_xlabel(self.config.x_label)
        self.ax.set_ylabel(self.config.y_label)
        self.ax.grid(True, linestyle=":", alpha=0.6)

    def transform_data(self):
        content = self.config.content

        x = np.arange(0, 1000)

        base_label = content.baseline.label
        baseline_path = content.baseline.data
        base_df = get_data(self.data_dir, baseline_path)
        base_df = calculate_accuracy_per_class(base_df)

        base_df = base_df.sort_values(ascending=False, by="accuracy")
        sorted_classes = base_df.index

        degraded_label = content.degraded.label
        degraded_path = content.degraded.data
        degraded_df = get_data(self.data_dir, degraded_path)
        degraded_df = calculate_accuracy_per_class(degraded_df)
        degraded_df = degraded_df.loc[sorted_classes]

        series = {}
        series[base_label] = base_df["accuracy"].values
        series[degraded_label] = degraded_df["accuracy"].values

        return {"x": x, "y": series}

    def render(self, data):
        x = data["x"]
        y = data["y"]
        labels = list(y.keys())

        x_ticks = list(range(0, 1000, 100))
        x_ticks.append(999)
        self.ax.set_xticks(x_ticks)

        self.ax.set_xlim([0, 1000])
        self.ax.set_ylim(0, 1.05)

        base = y[labels[0]]
        degraded = y[labels[1]]

        fragile_mask = (base >= 0.8) & (degraded <= 0.5)
        normal_mask = ~fragile_mask

        self.ax.plot(x, base, color="black", linewidth=2, label=labels[0])
        self.ax.scatter(
            x[normal_mask],
            degraded[normal_mask],
            color="#1f77b4",
            alpha=0.6,
            label=labels[1],
            zorder=3,
            edgecolors="none",
        )
        self.ax.scatter(
            x[fragile_mask],
            degraded[fragile_mask],
            color="red",
            alpha=0.6,
            label="Fragile classes",
            zorder=3,
            edgecolors="none",
        )

        self.ax.legend(title="Series", bbox_to_anchor=(1.01, 1.01), loc="upper left")
