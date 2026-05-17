from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


from plots.base import BasePlotPipeline

logger = logging.getLogger(__name__)


def _get_data(filename: str):
    path = Path("analysis/results") / filename

    with open(path, "r") as f:
        data = json.load(f)

    return pd.json_normalize(data, record_path=["classes"], meta=["name"])


class SpearmanRankPlot(BasePlotPipeline):
    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title(self.config.title, pad=75)

    def transform_data(self):
        content = self.config.content

        labels = [item["name"] for item in content]
        vectors = np.array(
            [_get_data(entry["data"])["accuracy_diff"].values for entry in content]
        )

        corr_matrix, _ = spearmanr(vectors, axis=1)
        corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)

        return corr_df

    def render(self, data: pd.DataFrame):
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)
        sns.heatmap(
            data,
            mask=mask,
            ax=self.ax,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Spearman's Rank Correlation (ρ)"},
        )
        self.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
