from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plots.base import BasePlotPipeline
from plots.data import get_data, calculate_accuracy_per_class


class DomainJaccardOverlapPlot(BasePlotPipeline):
    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.set_title(self.config.title, pad=20, fontsize=14)
        self.ax.grid(False)

    def _get_classes_set(
        self, corrupted_acc: pd.DataFrame, top_k: int, tail: str
    ) -> set:
        if tail == "best":
            sorted_acc = corrupted_acc.sort_values(
                by=["accuracy", "y_true"], ascending=[False, True]
            )
        else:
            sorted_acc = corrupted_acc.sort_values(
                by=["accuracy", "y_true"], ascending=[True, True]
            )

        k_classes = sorted_acc.head(top_k)
        return set(k_classes["y_true"].values)

    def transform_data(self):
        content = self.config.content
        models = content.models
        top_k = getattr(content, "top_k", 50)
        tail = getattr(content, "tail", "worst")

        labels = []
        worst_sets = []

        for model in models:
            labels.append(model.name)

            corrupted_df = get_data(self.data_dir, model.corrupted)
            corrupted_acc = calculate_accuracy_per_class(corrupted_df)

            worst_set = self._get_classes_set(corrupted_acc, top_k, tail)
            worst_sets.append(worst_set)

        n = len(labels)
        jaccard_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                set_i = worst_sets[i]
                set_j = worst_sets[j]

                intersection_size = len(set_i.intersection(set_j))
                union_size = len(set_i.union(set_j))

                jaccard_index = (
                    intersection_size / union_size if union_size > 0 else 1.0
                )
                jaccard_matrix[i, j] = jaccard_index

        jaccard_df = pd.DataFrame(jaccard_matrix, index=labels, columns=labels)
        return jaccard_df

    def render(self, data: pd.DataFrame):
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)

        sns.heatmap(
            data,
            mask=mask,
            ax=self.ax,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Jaccard Index", "alpha": 0.55},
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
        plt.yticks(rotation=0)
