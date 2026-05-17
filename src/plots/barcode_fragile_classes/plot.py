from __future__ import annotations

import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plots.base import BasePlotPipeline
from plots.data import get_data, calculate_accuracy_per_class
from model import MODELS

logger = logging.getLogger(__name__)

COMMON_THRESHOLD = 15


class BarcodeFragileClassesFreq(BasePlotPipeline):
    def _setup_canvas(self):
        num_models = len([m for m in MODELS.keys() if m != "alexnet"])
        self.fig, self.ax = plt.subplots(figsize=(20, max(8, num_models * 0.6)))
        self.fig.patch.set_facecolor("white")

    def transform_data(self) -> pd.DataFrame:
        content = self.config.content
        group_name = content.group
        corruptions_filter = content.corruptions
        severities_filter = content.severities

        from space import CorruptionVariations

        vs = CorruptionVariations(
            groups=[group_name],
            corruptions=corruptions_filter,
            severities=severities_filter,
        )
        variants = list({(v.corruption, v.severity) for v in vs})

        models = [m for m in MODELS.keys() if m != "alexnet"]
        plot_data = []

        for model_name in models:
            logger.info(f"Loading data for model: {model_name}")
            try:
                df_clean = get_data(self.data_dir, f"{model_name}_imagenet.csv")
            except FileNotFoundError:
                logger.warning(f"Skipping {model_name}: clean results not found")
                continue

            clean_acc = calculate_accuracy_per_class(df_clean).set_index("y_true")[
                "accuracy"
            ]

            corrupted_accs = []
            for corruption, severity in variants:
                filename = (
                    f"{model_name}_imagenet_c_{group_name}_{corruption}_{severity}.csv"
                )
                try:
                    df_corr = get_data(self.data_dir, filename)
                    acc = calculate_accuracy_per_class(df_corr).set_index("y_true")[
                        "accuracy"
                    ]
                    corrupted_accs.append(acc)
                except FileNotFoundError:
                    logger.warning(f"Missing: {filename}")

            if not corrupted_accs:
                logger.warning(
                    f"Skipping {model_name}: no corrupted results for group {group_name}"
                )
                continue

            avg_corrupted = pd.concat(corrupted_accs, axis=1).mean(axis=1)
            is_fragile = ((clean_acc >= 0.80) & (avg_corrupted <= 0.50)).astype(int)
            plot_data.append(is_fragile.rename(MODELS[model_name]))

        if not plot_data:
            return pd.DataFrame()

        fragile_wide = pd.DataFrame(plot_data)
        fragile_counts = fragile_wide.sum(axis=0)
        common_classes = set(fragile_counts[fragile_counts >= COMMON_THRESHOLD].index)

        for series in plot_data:
            series[:] = [
                2 if series[i] == 1 and i in common_classes else series[i]
                for i in series.index
            ]

        return pd.DataFrame(plot_data)

    def render(self, data: pd.DataFrame):
        if data.empty:
            return

        cmap = ListedColormap(["#EBEBEB", "#2C6E9E", "#F3636F"])

        sns.heatmap(
            data,
            cmap=cmap,
            vmin=0,
            vmax=2,
            cbar=False,
            ax=self.ax,
            xticklabels=False,
            yticklabels=True,
            linewidths=0,
        )

        for y in range(1, len(data)):
            self.ax.axhline(y, color="white", linewidth=0.8)

        plt.setp(
            self.ax.get_yticklabels(),
            fontfamily="monospace",
            fontsize=12,
            rotation=0,
            color="#222222",
        )

        n_classes = data.shape[1]
        tick_step = 25
        tick_positions = list(range(0, n_classes, tick_step)) + [n_classes - 1]
        self.ax.set_xticks([p + 0.5 for p in tick_positions])
        self.ax.set_xticklabels(
            [str(p) for p in tick_positions],
            fontsize=11,
            color="#555555",
            rotation=0,
        )

        self.ax.set_xlabel("ImageNet classes", fontsize=13, color="#444444", labelpad=6)
        self.ax.set_ylabel("")
        self.ax.tick_params(left=False)
        self.ax.spines[:].set_visible(False)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["bottom"].set_color("#aaaaaa")
        self.ax.tick_params(axis="x", length=4, width=0.8, color="#aaaaaa", bottom=True)
        self.ax.set_ylim(len(data) + 0.1, -0.1)

        plt.tight_layout()
