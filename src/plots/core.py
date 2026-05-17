from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from .base import BasePlotPipeline
from .specs import ChartConfig
from .accuracy_scatter_plot.plot import AccuracyToAccuracy, AccuracyToAccuracyDrop
from .fragile_class_similarity_matrix.plot import FragileClassSimilarityMatrix
from .class_degradation_plot.plot import SortedIndexClassDegradation
from .barcode_fragile_classes.plot import BarcodeFragileClassesFreq
from .spearman_rank_plot.plot import SpearmanRankPlot
from .violin_plot.plot import ViolinPlot

__all__ = ["create_plot"]

logger = logging.getLogger(__name__)


RENDERERS_PIPELINES: dict[str, type[BasePlotPipeline]] = {
    "accuracy_to_accuracy": AccuracyToAccuracy,
    "accuracy_to_accuracy_drop": AccuracyToAccuracyDrop,
    "sorted_index_class_degradation": SortedIndexClassDegradation,
    "barcode_fragile_classes": BarcodeFragileClassesFreq,
    "fragile_class_similarity_matrix": FragileClassSimilarityMatrix,
    "spearman_rank_plot": SpearmanRankPlot,
    "violin": ViolinPlot,
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
