from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import yaml


__all__ = ["get_plot_specs", "Axis", "ChartConfig"]

logger = logging.getLogger(__name__)


def get_plot_specs(config_path: Union[str, Path]) -> Iterator[ChartConfig]:
    plot_specs_path = Path(config_path)

    if not plot_specs_path.exists():
        logger.error(f"[ERROR] Plot specs file not found at: {plot_specs_path}")
        raise FileNotFoundError(
            f"Required plot specs file {plot_specs_path} is missing!"
        )

    return _PlotSpecsFactory.from_yaml(plot_specs_path)


@dataclass(frozen=True)
class Axis:
    label: str
    data: Optional[Union[str, list[str]]] = field(default=None)
    recipe: Optional[str] = field(default=None)
    values: Optional[list[Any]] = field(default=None)
    column: Optional[str] = field(default=None)
    operation: Optional[str] = field(default=None)
    lim: Optional[list[float]] = field(default=None)


@dataclass(frozen=True)
class ChartConfig:
    name: str
    type: str
    title: str
    x: Axis
    y: Optional[Axis] = field(default=None)
    y_series: Optional[list[Axis]] = field(default=None)
    output: str
    aux_line: Optional[str] = field(default=None)


class _PlotSpecsFactory:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[ChartConfig]:
        with open(yaml_path, "r") as f:
            contents = yaml.safe_load(f)

        return _PlotSpecsFactory._from_dict(contents)

    @staticmethod
    def _from_dict(content: Dict[str, Any]) -> Iterator[ChartConfig]:
        plots = content.get("plots", [])

        for plot in plots:
            x = _PlotSpecsFactory._resolve_axis(plot.get("x"))
            y = _PlotSpecsFactory._resolve_axis(plot.get("y"))
            y_series = (
                list(map(_PlotSpecsFactory._resolve_axis, plot.get("y_series")))
                if plot.get("y_series")
                else None
            )

            yield ChartConfig(
                name=plot.get("name"),
                type=plot.get("type"),
                title=plot.get("title"),
                x=x,
                y=y,
                y_series=y_series,
                output=plot.get("output"),
                aux_line=plot.get("aux_line"),
            )

    @staticmethod
    def _resolve_axis(axis: Dict[str, Any]) -> Optional[Axis]:
        if not axis:
            return None
        return Axis(**axis)
