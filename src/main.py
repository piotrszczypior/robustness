from __future__ import annotations

import os
import logging
import argparse
from typing import Dict

from paths import paths
from task import Task

import setup
import evaluate
import plots
import analyze
import xai
import sankey
import adversarial
import fragile
import plots_v2
from plots_v2.barcode.cli import get_task as get_barcode_v2_task
from plots_v2.acc_to_acc.cli import get_task as get_acc_to_acc_v2_task
from plots_v2.violin.cli import get_task as get_violin_v2_task
from plots_v2.rmce_heatmap.cli import get_task as get_rmce_heatmap_v2_task
from plots_v2.dataset_heatmap.cli import get_task as get_dataset_heatmap_task
from plots_v2.adversarial_dot.cli import get_task as get_adversarial_dot_v2_task
from plots_v2.class_degradation.cli import get_task as get_class_degradation_v2_task
from plots_v2.severity_dot.cli import get_task as get_severity_dot_v2_task
from plots_v2.fragile_histogram.cli import get_task as get_fragile_histogram_task
from plots_v2.upset_fragile.cli import get_task as get_upset_fragile_task
from plots_v2.fragile_severity_line.cli import get_task as get_fragile_severity_line_task
from plots_v2.mistake_dot.cli import get_task as get_mistake_dot_task
from plots_v2.fisher_heatmap.cli import get_task as get_fisher_heatmap_task
from plots_v2.jaccard.cli import get_task as get_jaccard_task
from plots_v2.fragile_dot import get_task as get_fragile_dot
from plots_v2.model_dot import get_task as get_model_dot
from plots_v2.mistake_models_dot import get_task as get_mistake_models_dot_task
from plots_v2.synset_model_dot import get_task as get_synset_model_dot_task
from corruptions.cli import get_task as get_corruptions_task
from representations.cli import get_task as get_representations_task



TASK_REGISTRY: Dict[str, Task] = {
    "setup": setup.get_task(),
    "evaluate": evaluate.get_task(),
    "plot": plots.get_task(),
    "analyze": analyze.get_task(),
    "xai": xai.get_task(),
    "sankey": sankey.get_task(),
    "adversarial": adversarial.get_task(),
    "fragile": fragile.get_task(),
    "spearman_v2": plots_v2.get_task(),
    "barcode_v2": get_barcode_v2_task(),
    "acc_to_acc_v2": get_acc_to_acc_v2_task(),
    "violin_v2": get_violin_v2_task(),
    "rmce_heatmap_v2": get_rmce_heatmap_v2_task(),
    "dataset_heatmap": get_dataset_heatmap_task(),
    "adversarial_dot_v2": get_adversarial_dot_v2_task(),
    "class_degradation_v2": get_class_degradation_v2_task(),
    "severity_dot_v2": get_severity_dot_v2_task(),
    "fragile_histogram_v2": get_fragile_histogram_task(),
    "upset_fragile_v2": get_upset_fragile_task(),
    "fragile_severity_line_v2": get_fragile_severity_line_task(),
    "mistake_dot_v2": get_mistake_dot_task(),
    "fisher_heatmap": get_fisher_heatmap_task(),
    "jaccard_v2": get_jaccard_task(),
    "corruptions": get_corruptions_task(),
    "representations": get_representations_task(),
    "fragile_dot_v2": get_fragile_dot(),
    "model_dot_v2": get_model_dot(),
    "mistake_models_dot_v2": get_mistake_models_dot_task(),
    "synset_model_dot_v2": get_synset_model_dot_task(),
}


def setup_logging():
    os.makedirs(paths.logs, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-35s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(paths.log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Robustness")
    subparsers = parser.add_subparsers(dest="task", required=True)

    for task in TASK_REGISTRY.values():
        task.register(subparsers)

    return parser.parse_args()


def main() -> int:
    logger = setup_logging()

    try:
        args = get_args()
        logger.info(f"Task '{args.task}' initialization")

        task = TASK_REGISTRY.get(args.task)
        if not task:
            logger.error(f"[ERROR] Task '{args.task}' not found in registry.")
            return 1

        task.run(args)
        logger.info(f"Task '{args.task}' completed")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Task failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main()
