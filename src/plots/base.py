import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from .specs import ChartConfig


from pathlib import Path


class BasePlotPipeline(ABC):
    def __init__(self, config: ChartConfig, data_dir: Path | str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.fig, self.ax = None, None
        self.data: dict[str, pd.DataFrame] = {}

    def run(self):
        self._validate()
        plot_data = self.transform_data()
        self._setup_canvas()
        self.render(plot_data)
        self._save()

    def schema(self) -> dict[str, type]:
        return {}

    def _validate(self):
        _ = self.schema()
        pass

    def _setup_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_title(self.config.title)
        self.ax.set_xlabel(self.config.x_label)
        self.ax.set_ylabel(self.config.y_label)
        self.ax.grid(True, linestyle=":", alpha=0.6)

    def _save(self):
        plt.tight_layout()
        self.fig.savefig(self.config.output)
        plt.close(self.fig)

    @abstractmethod
    def transform_data(self) -> Any:
        pass

    @abstractmethod
    def render(self, data: Any):
        pass
