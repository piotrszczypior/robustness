from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterator, Union

import yaml

__all__ = ["get_tasks"]


logger = logging.getLogger(__name__)


def get_tasks(config_path: Union[str, Path]) -> Iterator[BaseAnalysisConfig]:
    specs_path = Path(config_path)

    if not specs_path.exists():
        logger.error("")
        raise FileNotFoundError()

    return _SpecsLoader.from_yaml(specs_path)


@dataclass(frozen=True)
class BaseAnalysisConfig:
    name: str
    type: str
    content: Any


class _SpecsLoader:
    @staticmethod
    def from_yaml(yaml_path: Union[str, Path]) -> Iterator[BaseAnalysisConfig]:
        with open(yaml_path, "r") as f:
            contents = yaml.safe_load(f)

        analysis_tasks = contents.get("analyses", [])
        for task in analysis_tasks:
            yield BaseAnalysisConfig(**task)
