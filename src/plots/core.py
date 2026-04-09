from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from .base import BasePlotPipeline
from .specs import ChartConfig
from .plot import (
    AccuracyToAccuracy,
    AccuracyToAccuracyDrop,
    ClassDegradation,
    CompareFragileClasses,
    CompareFragileClasses2Pages,
    CompareFragileClassesFreq,
)

__all__ = ["create_plot"]

logger = logging.getLogger(__name__)


RENDERERS_PIPELINES: dict[str, BasePlotPipeline] = {
    "accuracy_to_accuracy": AccuracyToAccuracy,
    "accuracy_to_drop": AccuracyToAccuracyDrop,
    "sorted_index": ClassDegradation,
    "fragile_class": CompareFragileClasses,
    "fragile_class_freq": CompareFragileClassesFreq
}


def create_plot(
    plot_config: ChartConfig, base_data_path: Union[str, Path], debug: bool = False
) -> None:
    pipeline_cls = RENDERERS_PIPELINES.get(plot_config.type)

    if not pipeline_cls:
        logger.error(f"[ERROR] Unsupported plot type: {plot_config.type}")
        raise ValueError(f"[ERROR] Unsupported plot type: {plot_config.type}")

    logger.info(
        f"Rendering plot '{plot_config.name}' using '{plot_config.type}' renderer"
    )

    pipeline_cls(config=plot_config, data_dir=base_data_path).run()
