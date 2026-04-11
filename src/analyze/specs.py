from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterator, Union

import yaml

__all__ = ["get_specs"]


logger = logging.getLogger(__name__)


def get_specs(config_path: Union[str, Path]) -> Iterator[AnalysisConfig]:
    specs_path = Path(config_path)

    if not specs_path.exists():
        logger.error("")
        raise FileNotFoundError()

    return _SpecsLoader.from_yaml(specs_path)


@dataclass(frozen=True)
class DataSource:
    baseline: str
    degraded: str


@dataclass(frozen=True)
class AnalysisConfig:
    name: str
    type: str
    data: DataSource


class _SpecsLoader:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[AnalysisConfig]:
        with open(yaml_path, "r") as f:
            contents = yaml.safe_load(f)

        analysis_tasks = contents.get("analyses", [])
        for task in analysis_tasks:
            task["data"] = DataSource(**task["data"])

            yield AnalysisConfig(**task)
