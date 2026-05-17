from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from plots.base import BasePlotPipeline

logger = logging.getLogger(__name__)


def _get_data(filename: str):
    path = Path("analysis/results") / filename

    with open(path, "r") as f:
        data = json.load(f)

    return pd.json_normalize(data, record_path=["classes"], meta=["name"])


class FragileClassSimilarityMatrix(BasePlotPipeline):
    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title(self.config.title)
        self.ax.set_xlabel(self.config.x_label)
        self.ax.set_ylabel(self.config.y_label)

    def transform_data(self):
        content = self.config.content

        labels = [entry["name"] for entry in content]
        vectors = np.array(
            [_get_data(entry["data"])["is_fragile"].values for entry in content]
        )

        dist = pdist(vectors, metric="jaccard")
        similarity_matrix = squareform(1 - dist)
        np.fill_diagonal(similarity_matrix, 1.0)

        return pd.DataFrame(similarity_matrix, index=labels, columns=labels)

    def render(self, data: pd.DataFrame):
        sns.heatmap(
            data,
            annot=True,
            cmap=sns.light_palette("seagreen", as_cmap=True),
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Jaccard Index"},
            ax=self.ax,
        )
        self.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
