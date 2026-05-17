from __future__ import annotations
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plots.base import BasePlotPipeline
from mce import (
    load_and_aggregate_results,
    aggregate_for_rmce,
    compute_rmce_mce,
    get_denom_indices,
)
from model import MODELS

logger = logging.getLogger(__name__)


class BarcodeRmCEPlot(BasePlotPipeline):
    def _setup_canvas(self):
        content = self.config.content
        num_models = len(content.models)
        figsize = (20, 12)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor("white")

    def transform_data(self) -> pd.DataFrame:
        content = self.config.content
        models = [m for m in MODELS.keys() if m != "alexnet"]
        group_name = content.group
        corruptions_filter = content.corruptions
        severities_filter = content.severities
        common_threshold = 15

        logger.info(f"Loading AlexNet baseline data for group {group_name}...")
        try:
            df_alexnet = load_and_aggregate_results("alexnet", self.data_dir)
        except Exception as e:
            logger.error(f"Failed to load AlexNet data: {e}")
            raise

        from space import CorruptionVariations

        vs = CorruptionVariations(
            groups=[group_name],
            corruptions=corruptions_filter,
            severities=severities_filter,
        )
        group_corruptions = list(set(v.corruption for v in vs))

        agg_alex = aggregate_for_rmce(
            df_alexnet, corruptions=group_corruptions, severities=severities_filter
        )
        stable_synsets_alexnet = get_denom_indices(agg_alex)

        plot_data = []  # list of series/dicts for each model

        for model_name in models:
            logger.info(f"Loading data for model: {model_name}")
            df_model = load_and_aggregate_results(model_name, self.data_dir)

            # if not group_corruptions:
            #     continue

            agg_model = aggregate_for_rmce(
                df_model, corruptions=group_corruptions, severities=severities_filter
            )
            rmce_df = compute_rmce_mce(agg_model, agg_alex)

            # Create boolean column
            rmce_df["is_fragile"] = (
                rmce_df["synset"].isin(stable_synsets_alexnet) & (rmce_df["RmCE"] > 2)
            ).astype(int)

            # Ensure synsets are sorted or at least consistent
            rmce_df = rmce_df.sort_values("synset")

            # Map values
            model_data = rmce_df.set_index("synset")["is_fragile"].rename(
                MODELS[model_name]
            )
            plot_data.append(model_data)

        if not plot_data:
            return pd.DataFrame()

        fragile_wide = pd.DataFrame(plot_data)
        fragile_counts = fragile_wide.sum(axis=0)
        common_synsets = set(fragile_counts[fragile_counts >= common_threshold].index)

        def level(synset):
            if series[synset] == 1 and synset in common_synsets:
                return 2
            return series[synset]

        for series in plot_data:
            series[:] = series.index.map(level)

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
            fontsize=14,
            rotation=0,
            color="#222222",
        )

        n_classes = data.shape[1]
        tick_step = 25
        tick_positions = list(range(0, n_classes, tick_step)) + [999]
        self.ax.set_xticks([p + 0.5 for p in tick_positions])
        self.ax.set_xticklabels(
            [str(p) for p in tick_positions],
            fontsize=12,
            color="#555555",
            rotation=0,
        )

        self.ax.set_xlabel(
            "ImageNet classes",
            fontsize=14,
            color="#444444",
            labelpad=6,
        )
        self.ax.set_ylabel("")
        self.ax.tick_params(left=False)
        self.ax.spines[:].set_visible(False)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["bottom"].set_color("#aaaaaa")
        self.ax.tick_params(axis="x", length=4, width=0.8, color="#aaaaaa", bottom=True)
        self.ax.set_ylim(len(data) + 0.1, -0.1)

        # from matplotlib.patches import Patch
        # legend_elements = [
        #     Patch(facecolor="#EBEBEB", edgecolor="#cccccc", label="Robust"),
        #     Patch(facecolor="#2C6E9E", label="Fragile"),
        # ]
        # self.ax.legend(
        #     handles=legend_elements,
        #     loc="upper left",
        #     bbox_to_anchor=(1.01, 1),
        #     borderaxespad=0,
        #     fontsize=10,
        #     framealpha=0.9,
        #     edgecolor="#cccccc",
        # )

        # self.fig.suptitle(
        #     self.config.title,
        #     fontsize=13, fontweight="bold", color="#111111", y=1.02,
        # )
        plt.tight_layout()
