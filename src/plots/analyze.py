from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .recipe import get_recipe
from .specs import ChartConfig

__all__ = ["create_plot"]

logger = logging.getLogger(__name__)


def create_plot(plot_config: ChartConfig, base_results_path: Union[str, Path]) -> None:
    return _PlotFactory.create(plot_config, Path(base_results_path))


@dataclass(frozen=True)
class PlotContext:
    x: pd.Series
    y: pd.Series
    title: str
    x_label: str
    y_label: str
    output_path: Path


class _PlotFactory:
    @staticmethod
    def create(config: ChartConfig, results_dir: Path) -> None:
        x_path = results_dir / config.x.data
        y_path = results_dir / config.y.data

        if not _DataLoader.exists(x_path, y_path):
            logger.warning(f"Skipping plot '{config.name}': Data files missing.")
            return

        x_df = _DataLoader.load(x_path)
        y_df = _DataLoader.load(y_path)

        recipe = get_recipe(config.recipe)

        context = PlotContext(
            x=recipe.apply(x_df),
            y=recipe.apply(y_df),
            title=config.title,
            x_label=config.x.label,
            y_label=config.y.label,
            output_path=Path(config.output),
        )

        _PlotRenderer.render(recipe.type, context)


class _DataLoader:
    @staticmethod
    def exists(*paths: Path) -> bool:
        for p in paths:
            if not p.exists():
                logger.debug(f"File not found: {p}")
                return False
        return True

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)


class _PlotRenderer:
    @staticmethod
    def render(plot_type: str, context: PlotContext) -> None:
        renderers = {
            "scatter": _PlotRenderer._scatter,
        }

        renderer = renderers.get(plot_type)
        if not renderer:
            logger.error(f"Unsupported plot type: {plot_type}")
            return

        renderer(context)

    @staticmethod
    def _scatter(context: PlotContext) -> None:
        data = pd.concat([context.x, context.y], axis=1, join="inner")
        data.columns = ["x", "y"]

        if data.empty:
            logger.warning(f"No data points for '{context.title}'.")
            return

        plt.figure(figsize=(10, 10))
        sns.scatterplot(data=data, x="x", y="y", alpha=0.5)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", alpha=0.7)

        plt.title(context.title)
        plt.xlabel(context.x_label)
        plt.ylabel(context.y_label)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)

        os.makedirs(context.output_path.parent, exist_ok=True)
        plt.savefig(context.output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot '{context.title}' saved to {context.output_path}")
