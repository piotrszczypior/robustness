from __future__ import annotations

import logging

import pandas as pd
import seaborn as sns

from plots.base import BasePlotPipeline
from plots.data import get_data, calculate_accuracy_per_class
from subsets import filder_imagenet_r_classes

logger = logging.getLogger(__name__)


class AccuracyToAccuracy(BasePlotPipeline):
    def schema(self):
        return {"x": str, "y": str}

    def transform_data(self):
        content = self.config.content

        x = get_data(self.data_dir, content.x)
        y = get_data(self.data_dir, content.y)

        x = calculate_accuracy_per_class(x)
        y = calculate_accuracy_per_class(y)

        print(x.head(5))
        print(len(x))

        x = filder_imagenet_r_classes(x)

        print(x.head(5))
        print(len(x))
        print(len(y))

        print(y.head(5))

        data = pd.merge(x, y, on="y_true")
        data = data.drop(columns=["y_true"])
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

        x = calculate_accuracy_per_class(x)
        y = calculate_accuracy_per_class(y)

        y = y - x

        data = pd.merge(x, y, left_index=True, right_index=True)
        data.columns = ["x", "y"]

        return data

    def render(self, data):
        self.ax.set_ylim([-1, 1])
        sns.scatterplot(data=data, x="x", y="y", ax=self.ax, alpha=0.5)
        self.ax.axhline(0, color="red", linestyle="--", alpha=0.7)
