from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .recipe import get_recipe
from .specs import ChartConfig, Axis

__all__ = ["create_plot"]

logger = logging.getLogger(__name__)


def create_plot(
    plot_config: ChartConfig, base_results_path: Union[str, Path], debug: bool = False
) -> None:
    return _PlotFactory.create(plot_config, Path(base_results_path), debug)


@dataclass(frozen=True)
class PlotContext:
    x: pd.Series
    y: pd.Series
    title: str
    aux_line: str
    x_label: str
    y_label: str
    output_path: Path
    x_operation: Optional[str] = None
    y_operation: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    x_lim: list[float] = [0, 1.05]
    y_lim: list[float] = [0, 1.05]


class _PlotFactory:
    @staticmethod
    def create(config: ChartConfig, results_dir: Path, debug: bool = False) -> None:
        x = _PlotFactory._get_series(results_dir, config.x)
        y = _PlotFactory._get_series(results_dir, config.y)

        if x.empty or y.empty:
            logger.warning(f"Skipping plot '{config.name}': Data missing.")
            return

        if config.x and config.x.operation == "diff":
            x = y - x
        if config.y and config.y.operation == "diff":
            y = y - x

        context = PlotContext(
            x=x,
            y=y,
            title=config.title,
            x_label=config.x.label,
            y_label=config.y.label,
            output_path=Path(config.output),
            x_operation=config.x.operation,
            y_operation=config.y.operation,
            x_column="x" if config.x.column is None else config.x.column,
            y_column="y" if config.y.column is None else config.y.column,
            x_lim=[0, 1.05] if config.x.lim is None else config.x.lim,
            y_lim=[0, 1.05] if config.y.lim is None else config.y.lim,
            aux_line="diagonal" if config.aux_line is None else config.aux_line,
        )

        _PlotRenderer.render(config.type, context, debug)

    @staticmethod
    def _get_series(results_dir: Path, axis: Axis) -> pd.Series:
        if axis.values is not None:
            return pd.Series(axis.values)

        if axis.data is None:
            return pd.Series(dtype=float)

        file_list = [axis.data] if type(axis.data) is str else axis.data

        series_list = []
        for file_name in file_list:
            path = results_dir / file_name
            series_list.append(_PlotFactory._load_file(path, axis.recipe))

        if not series_list:
            return pd.Series(dtype=float)

        return pd.concat(series_list, ignore_index=True)

    @staticmethod
    def _load_file(file_path: str, recipe_name: str) -> pd.Series:
        df = _DataLoader.load(Path(file_path))
        return _PlotFactory._apply_recipe(df, recipe_name)

    @staticmethod
    def _apply_recipe(df: pd.DataFrame, recipe_name: str) -> pd.Series:
        recipe = get_recipe(recipe_name)
        return recipe.apply(df)


class _DataLoader:
    @staticmethod
    def exists(*paths: Path) -> bool:
        for p in paths:
            if not p.exists():
                logger.error(f"File not found: {p}")
                return False
        return True

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        logger.info(f"Loading data from {path}")
        if not _DataLoader.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        return pd.read_csv(path)


class _PlotRenderer:
    @staticmethod
    def render(plot_type: str, context: PlotContext, debug: bool) -> None:
        renderers = {"scatter": _PlotRenderer._scatter, "bar": _PlotRenderer._bar}

        renderer = renderers.get(plot_type)
        if not renderer:
            logger.error(f"[ERROR] Unsupported plot type: {plot_type}")
            raise ValueError(f"Unsupported plot type: {plot_type}")

        if debug:
            logger.debug(f"[DEBUG]: {context.title}")
            logger.debug(f"Plot type: {plot_type}")
            logger.debug(f"X label ({context.x_label})")
            logger.debug(f"X series: \n {context.x.head(3)}")
            logger.debug(f"Y label ({context.y_label})")
            logger.debug(f"Y series: \n {context.y.head(3)}")
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

        if context.aux_line == "diagonal":
            plt.plot([0, 0], [1, 0], color="red", linestyle="--", alpha=0.7)
        elif context.aux_line == "y=0":
            plt.axhline(0, color="red", linestyle="--", alpha=0.7)

        plt.title(context.title)
        plt.xlabel(context.x_label)
        plt.ylabel(context.y_label)

        plt.xlim(context.x_lim[0], context.x_lim[1])
        plt.ylim(context.y_lim[0], context.y_lim[1])

        plt.grid(True, linestyle=":", alpha=0.6)

        os.makedirs(context.output_path.parent, exist_ok=True)
        if context.output_path.exists():
            logger.info(
                f"[SKIP] Plot {context.title} already exists. Skipping saving..."
            )
            return

        plt.savefig(context.output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot '{context.title}' saved to {context.output_path}")

    @staticmethod
    def _bar(context: PlotContext) -> None:
        data = pd.concat([context.x, context.y], axis=1, join="inner")
        data.columns = ["x", "y"]

        if data.empty:
            logger.warning(f"No data points for '{context.title}'.")
            return

        plt.figure(figsize=(10, 10))

        plt.title(context.title)
        plt.xlabel(context.x_label)
        plt.ylabel(context.y_label)

        sns.barplot(data=data, x=context.x_column, y=context.y_column)

        plt.grid(True, linestyle=":", alpha=0.6)

        os.makedirs(context.output_path.parent, exist_ok=True)
        if context.output_path.exists():
            logger.info(
                f"[SKIP] Plot {context.title} already exists. Skipping saving..."
            )
            return

        plt.savefig(context.output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot '{context.title}' saved to {context.output_path}")
