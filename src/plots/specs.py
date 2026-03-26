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
    data: str
    recipe: str
    column: Optional[str] = field(default=None)
    operation: Optional[str] = field(default=None)


@dataclass(frozen=True)
class ChartConfig:
    name: str
    type: str
    title: str
    x: Optional[Axis]
    y: Optional[Axis]
    output: str


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
            name = plot.get("name")
            type = plot.get("type")
            title = plot.get("title")
            output = plot.get("output")

            x = _PlotSpecsFactory._resolve_axis(plot.get("x"))
            y = _PlotSpecsFactory._resolve_axis(plot.get("y"))

            yield ChartConfig(name, type, title, x, y, output)

    @staticmethod
    def _resolve_axis(axis: Dict[str, Any]) -> Optional[Axis]:
        if not axis:
            return None
        return Axis(**axis)
