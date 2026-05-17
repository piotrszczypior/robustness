from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
from plots.base import BasePlotPipeline
from plots.data import get_data, calculate_accuracy_per_class


class DomainSpearmanRankPlot(BasePlotPipeline):
    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.set_title(self.config.title, pad=20, fontsize=14, fontweight="bold")
        self.ax.grid(False)

    def _get_drop_vector(
        self, clean_acc: pd.DataFrame, corrupted_acc: pd.DataFrame
    ) -> np.ndarray:
        combined = pd.merge(
            clean_acc, corrupted_acc, on="y_true", suffixes=("_clean", "_corrupted")
        )
        # FIXME: changed to relative drop 
        drop = (combined["accuracy_clean"] - combined["accuracy_corrupted"])
        return drop.values

    def _get_rank_vector(self, corrupted_acc: pd.DataFrame) -> np.ndarray:
        sorted_acc = corrupted_acc.sort_values(
            by=["accuracy", "y_true"], ascending=[False, True]
        )
        sorted_acc["rank"] = np.arange(1, len(sorted_acc) + 1)
        sorted_back = sorted_acc.sort_values(by="y_true")
        return sorted_back["rank"].values

    def _get_averaged_drop_vector(self, model_config) -> np.ndarray:
        clean_df = get_data(self.data_dir, model_config.clean)
        clean_acc = calculate_accuracy_per_class(clean_df)

        all_drop_vectors = []
        for corrupted_file in model_config.corrupted_files:
            try:
                corrupted_df = get_data(self.data_dir, corrupted_file)
                corrupted_acc = calculate_accuracy_per_class(corrupted_df)
                drop_vector = self._get_drop_vector(clean_acc, corrupted_acc)
                all_drop_vectors.append(drop_vector)
            except FileNotFoundError:
                # logger.warning(f"File not found: {corrupted_file}, skipping.")
                continue

        if not all_drop_vectors:
            return np.array([])

        averaged_drop_vector = np.mean(all_drop_vectors, axis=0)
        return averaged_drop_vector

    def transform_data(self):
        content = self.config.content
        models = content.models
        metric_type = getattr(content, "metric_type", "drop")
        is_averaged = getattr(content, "is_averaged", False)

        labels = []
        vectors = []

        for model in models:
            labels.append(model.name)

            if is_averaged:
                if metric_type == "drop":
                    vector = self._get_averaged_drop_vector(model)
                else:
                    # logger.warning(
                    #     "Averaging is only supported for 'drop' metric. Skipping."
                    # )
                    vector = np.array([])
            else:
                clean_df = get_data(self.data_dir, model.clean)
                corrupted_df = get_data(self.data_dir, model.corrupted)

                clean_acc = calculate_accuracy_per_class(clean_df)
                corrupted_acc = calculate_accuracy_per_class(corrupted_df)

                if metric_type == "rank":
                    vector = self._get_rank_vector(corrupted_acc)
                else:
                    vector = self._get_drop_vector(clean_acc, corrupted_acc)

            if vector.size > 0:
                vectors.append(vector)

        if not vectors:
            return pd.DataFrame()

        vectors = np.array(vectors)
        corr_matrix, _ = spearmanr(vectors, axis=1)

        if np.isscalar(corr_matrix):
            corr_matrix = np.array([[corr_matrix]])

        corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
        return corr_df

    def render(self, data: pd.DataFrame):
        # mask = np.triu(np.ones_like(data, dtype=bool), k=1)

        bounds = np.arange(0.4, 1.05, 0.05)
        # cmap = sns.color_palette("coolwarm", n_colors=10)

        base_cmap = plt.get_cmap("coolwarm")
        cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, len(bounds) - 1)))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        sns.heatmap(
            data,
            # mask=mask,
            ax=self.ax,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            norm=norm,
            vmin=0.3,
            vmax=0.9,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Spearman's Rank Correlation (ρ)", "alpha": 0.55},
        )

        self.ax.tick_params(
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=True,
        )
        plt.xticks(rotation=90)
        plt.xticks(rotation=60, ha="left")
