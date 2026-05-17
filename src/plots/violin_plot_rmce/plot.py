from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plots.base import BasePlotPipeline
from mce import load_and_aggregate_results, aggregate_for_rmce, compute_rmce_mce

logger = logging.getLogger(__name__)


class ViolinRmCEPlot(BasePlotPipeline):
    def _setup_canvas(self):
        content = self.config.content
        n = len(content.models)
        self.fig, self.ax = plt.subplots(figsize=(11, max(3, n * 1.4)))
        self.ax.set_title(self.config.title, pad=16, fontsize=13, fontweight="bold")

    def transform_data(self) -> dict[str, np.ndarray]:
        content = self.config.content
        models = content.models
        corruptions_filter = content.corruptions
        severities_filter = content.severities

        logger.info("Loading AlexNet baseline data...")
        try:
            df_alexnet = load_and_aggregate_results("alexnet", self.data_dir)
        except Exception as e:
            logger.error(f"Failed to load AlexNet data: {e}")
            raise

        from space import CorruptionVariations

        plot_data: dict[str, np.ndarray] = {}

        for model_name in models:
            logger.info(f"Loading data for model: {model_name}")
            df_model = load_and_aggregate_results(model_name, self.data_dir)

            vs = CorruptionVariations(
                corruptions=corruptions_filter,
                severities=severities_filter,
            )
            selected_corruptions = list(set(v.corruption for v in vs))

            agg_model = aggregate_for_rmce(
                df_model,
                "all",
                corruptions=selected_corruptions,
                severities=severities_filter,
            )
            agg_alex = aggregate_for_rmce(
                df_alexnet,
                "all",
                corruptions=selected_corruptions,
                severities=severities_filter,
            )
            rmce_df = compute_rmce_mce(agg_model, agg_alex, "all")

            plot_data[model_name] = rmce_df["RmCE"].dropna().values

        return plot_data

    def render(self, data: dict[str, np.ndarray]):
        ax = self.ax
        labels = list(data.keys())
        n = len(labels)
        color = "#5b8db8"

        for i, label in enumerate(labels):
            vals = data[label]
            if len(vals) == 0:
                continue

            pos = n - i  # top → bottom

            # --- violin ---
            parts = ax.violinplot(
                [vals],
                positions=[pos],
                widths=0.7,
                vert=False,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
                pc.set_edgecolor("none")

            # --- boxplot overlay ---
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
            whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
            outliers = vals[(vals < whisker_lo) | (vals > whisker_hi)]

            ax.plot(
                [whisker_lo, whisker_hi],
                [pos, pos],
                color="#333333",
                linewidth=1.0,
                zorder=3,
            )
            box = mpatches.FancyBboxPatch(
                (q1, pos - 0.12),
                iqr,
                0.24,
                boxstyle="square,pad=0",
                linewidth=1.2,
                edgecolor="#333333",
                facecolor="white",
                zorder=4,
            )
            ax.add_patch(box)
            ax.plot(
                [median, median],
                [pos - 0.12, pos + 0.12],
                color="#e07b39",
                linewidth=2.0,
                zorder=5,
            )
            if len(outliers):
                ax.scatter(
                    outliers,
                    np.full_like(outliers, pos),
                    s=12,
                    color="#333333",
                    alpha=0.5,
                    zorder=3,
                    linewidths=0,
                )

            # --- right-side label: mean ± std ---
            mean, std = vals.mean(), vals.std()
            ax.annotate(
                f"{label}\n({mean:.2f} ± {std:.2f})",
                xy=(1.01, pos),
                xycoords=("axes fraction", "data"),
                fontsize=9,
                va="center",
                color="#222222",
            )

        # --- AlexNet reference line ---
        ax.axvline(
            1.0,
            color="#c0392b",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
            label="AlexNet baseline",
        )

        ax.set_yticks(range(1, n + 1))
        ax.set_yticklabels(reversed(labels), fontsize=10)
        ax.set_xlabel("RmCE per class", fontsize=11)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="lower right")
