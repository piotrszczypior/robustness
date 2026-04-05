from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Union

import yaml
from munch import DefaultMunch

__all__ = ["get_plot_specs", "ChartConfig"]


logger = logging.getLogger(__name__)


def get_plot_specs(config_path: Union[str, Path]) -> Iterator[Any]:
    plot_specs_path = Path(config_path)

    if not plot_specs_path.exists():
        logger.error(f"[ERROR] Plot specs file not found at: {plot_specs_path}")
        raise FileNotFoundError(
            f"Required plot specs file {plot_specs_path} is missing!"
        )

    return _PlotSpecsFactory.from_yaml(plot_specs_path)


@dataclass(frozen=True)
class ChartConfig:
    name: str
    title: str
    x_label: str
    y_label: str
    type: str
    output: str
    content: Any


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
            content = plot.get("content", {})

            yield ChartConfig(
                name=plot.get("name"),
                title=plot.get("title"),
                type=plot.get("type"),
                x_label=plot.get("x_label"),
                y_label=plot.get("y_label"),
                output=plot.get("output"),
                content=DefaultMunch.fromDict(content),
            )
