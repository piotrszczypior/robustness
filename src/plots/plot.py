from __future__ import annotations

import logging
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .base import BasePlotPipeline
from .data import get_data

logger = logging.getLogger(__name__)


def _calculate_accuracy_per_class(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["y_true"])["is_correct"].agg(accuracy="mean").reset_index()


class AccuracyToAccuracy(BasePlotPipeline):
    def schema(self):
        return {"x": str, "y": str}

    def transform_data(self):
        content = self.config.content

        x = get_data(self.data_dir, content.x)
        y = get_data(self.data_dir, content.y)

        x = _calculate_accuracy_per_class(x)
        y = _calculate_accuracy_per_class(y)

        data = pd.merge(x, y, left_index=True, right_index=True)
        data.columns = ["x", "y"]

        return data

    def render(self, data):
        sns.scatterplot(data=data, x="x", y="y", ax=self.ax, alpha=0.5)
        self.ax.plot([0, 1.05], [0, 1.05], color="red", linestyle="--", alpha=0.7)


class AccuracyToAccuracyDrop(BasePlotPipeline):
    def schema(self):
        return {"x": str, "y": str}

    def transform_data(self):
        content = self.config.content

        x = get_data(self.data_dir, content.x)
        y = get_data(self.data_dir, content.y)

        x = _calculate_accuracy_per_class(x)
        y = _calculate_accuracy_per_class(y)

        y = y - x

        data = pd.merge(x, y, left_index=True, right_index=True)
        data.columns = ["x", "y"]

        return data

    def render(self, data):
        self.ax.set_ylim([-1, 1])
        sns.scatterplot(data=data, x="x", y="y", ax=self.ax, alpha=0.5)
        self.ax.axhline(0, color="red", linestyle="--", alpha=0.7)


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
        base_df = _calculate_accuracy_per_class(base_df)

        base_df = base_df.sort_values(ascending=False, by="accuracy")
        sorted_classes = base_df.index

        degraded_label = content.degraded.label
        degraded_path = content.degraded.data
        degraded_df = get_data(self.data_dir, degraded_path)
        degraded_df = _calculate_accuracy_per_class(degraded_df)
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


class CompareFragileClasses(BasePlotPipeline):
    def schema(self):
        return [{"label": str, "data": str}]

    def _setup_canvas(self):
        self.fig = plt.figure(figsize=(12, 6))
        self.gs = GridSpec(1, 2, figure=self.fig, wspace=0.3)

    def transform_data(self):
        content = self.config.content
        dfs = []
        for entry in content:
            df = self._get_data(entry["data"])
            df = df[["y_true", "is_fragile"]].rename(
                columns={"is_fragile": entry["name"], "y_true": "class_index"}
            )
            dfs.append(df.set_index("class_index"))

        return pd.concat(dfs, axis=1).sort_index()

    def render(self, data: pd.DataFrame):
        cmap = ListedColormap(["#F2F2F2", "#1F77B4"])
        col_num = data.shape[1]

        half = len(data) // 2
        parts = [data.iloc[:half], data.iloc[half:]]
        x_ticks = list(range(0, half, 25))
        x_ticks.append(half - 1)

        axes = [
            self.fig.add_subplot(self.gs[0, 0]),
            self.fig.add_subplot(self.gs[0, 1]),
        ]

        for i, (ax, part_data) in enumerate(zip(axes, parts)):
            sns.heatmap(
                part_data,
                cmap=cmap,
                cbar=False,
                ax=ax,
                xticklabels=True,
                yticklabels=True,
                linewidths=0,
            )

            indices = part_data.index
            ax.set_yticks(x_ticks)
            ax.set_yticklabels([indices[x] for x in x_ticks])

            ax.xaxis.tick_top()
            plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=0)

            for col_idx in range(1, col_num):
                ax.axvline(col_idx, color="white", linewidth=5)

            ax.set_ylabel("Class Index")
            ax.tick_params(axis="both", rotation=0)

        self.fig.suptitle(self.config.title, fontsize=14, fontweight="bold", y=0.98)

    def _get_data(self, filename: str):
        import json
        from pathlib import Path

        path = Path("analysis_results") / filename

        with open(path, "r") as f:
            data = json.load(f)

        return pd.json_normalize(data, record_path=["classes"], meta=["name"])


class CompareFragileClassesFreq(BasePlotPipeline):
    def schema(self):
        return [{"label": str, "data": str}]

    def _setup_canvas(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.gs = GridSpec(1, 2, figure=self.fig, wspace=0.3)

    def transform_data(self):
        content = self.config.content
        dfs = []
        for entry in content:
            df = self._get_data(entry["data"])
            df = df[["y_true", "is_fragile"]].rename(
                columns={"is_fragile": entry["name"], "y_true": "class_index"}
            )
            dfs.append(df.set_index("class_index"))

        return pd.concat(dfs, axis=1).sort_index()

    def render(self, data: pd.DataFrame):
        colors = ["#F2F2F2", "#1F77B4", "#00FF00", "#D62728"]
        cmap = ListedColormap(colors)
        col_num = data.shape[1]

        display_data = data.copy()
        row_sums = data.sum(axis=1)

        for idx in data.index:
            row_sum = row_sums.loc[idx]
            if row_sum == col_num:
                display_data.loc[idx] = data.loc[idx] * 3
            elif row_sum > 1:
                display_data.loc[idx] = data.loc[idx] * 2
            else:
                display_data.loc[idx] = data.loc[idx] * 1

        half = len(data) // 2
        parts = [display_data.iloc[:half], display_data.iloc[half:]]

        x_ticks = list(range(0, half, 25))
        x_ticks.append(half - 1)

        axes = [
            self.fig.add_subplot(self.gs[0, 0]),
            self.fig.add_subplot(self.gs[0, 1]),
        ]

        for i, (ax, part_data) in enumerate(zip(axes, parts)):
            sns.heatmap(
                part_data,
                cmap=cmap,
                cbar=False,
                ax=ax,
                xticklabels=True,
                yticklabels=True,
                linewidths=0,
                vmin=0,
                vmax=3,
            )

            indices = part_data.index
            ax.set_yticks(x_ticks)
            ax.set_yticklabels([indices[x] for x in x_ticks])

            ax.xaxis.tick_top()
            plt.setp(
                ax.get_xticklabels(),
                fontsize=8,
                fontweight="bold",
                rotation=35,
            )

            for col_idx in range(1, col_num):
                ax.axvline(col_idx, color="white", linewidth=5)

            ax.set_ylabel("Class Index")
            ax.tick_params(axis="y", rotation=0)

        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.25)
        self.fig.suptitle(self.config.title, fontsize=14, y=0.98)

    def _get_data(self, filename: str):
        import json
        from pathlib import Path

        path = Path("analysis/results") / filename

        with open(path, "r") as f:
            data = json.load(f)

        return pd.json_normalize(data, record_path=["classes"], meta=["name"])
