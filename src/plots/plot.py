from __future__ import annotations

import logging
import seaborn as sns
import pandas as pd
import numpy as np

from plot import BasePlotPipeline

from .data import get_dataframe
from .recipe import get_recipe

logger = logging.getLogger(__name__)


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    recipe = get_recipe("accuracy")
    return recipe.transform(df)


class AccuracyToAccuracy(BasePlotPipeline):
    def schema(self):
        return {"x": str, "y": str}

    def transform_data(self):
        content = self.config.content

        x = get_dataframe(self.data_dir, content.x)
        y = get_dataframe(self.data_dir, content.y)

        x = _transform(x)
        y = _transform(y)

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

        x = get_dataframe(self.data_dir, content.x)
        y = get_dataframe(self.data_dir, content.y)

        x = _transform(x)
        y = _transform(y)

        y = y - x

        data = pd.merge(x, y, left_index=True, right_index=True)
        data.columns = ["x", "y"]

        return data

    def render(self, data):
        self.ax.set_ylim([-1, 1])
        sns.scatterplot(data=data, x="x", y="y", ax=self.ax, alpha=0.5)
        self.ax.axhline(0, color="red", linestyle="--", alpha=0.7)


class ClassDegradation(BasePlotPipeline):
    def schema(self):
        return {"basefile": dict, "series": list}

    def transform_data(self):
        content = self.config.content

        x = np.arange(0, 1000)
        series = []

        base_label = content.basefile.label
        baseline_path = content.basefile.file
        base_df = get_dataframe(self.data_dir, baseline_path)
        base_df = _transform(base_df)

        sorted_classes = base_df.sort_values(ascending=False, by="accuracy").index

        for s in content.series:
            df = get_dataframe(self.data_dir, s.file)
            df = _transform(df)
            series.append((s.label, df.loc[sorted_classes].values.flatten()))

        data = {
            base_label: base_df.loc[sorted_classes].values.flatten(),
            **dict(series),
        }

        return {"x": x, "y": data}

    def render(self, data):
        x_ticks = [200, 400, 600, 800, 1000]
        self.ax.set_xticks(x_ticks)
        self.ax.set_xlim([0, 1000])

        window_size = 30
        x = data["x"]
        y_data = data["y"]

        for i, (label, y) in enumerate(y_data.items()):
            y_smooth = (
                pd.Series(y)
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            self.ax.plot(x, y_smooth, linewidth=2, label=label)

        self.ax.legend(title="Series", bbox_to_anchor=(1.01, 1.01), loc="upper left")
