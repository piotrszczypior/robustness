from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

from analyze.analyses import AccuracyDropTask

logger = logging.getLogger(__name__)


def run(task: AccuracyDropTask, output_dir: str) -> None:
    _AccuracyDropGenerator(task, output_dir).run()


class _AccuracyDropGenerator:
    def __init__(self, task: AccuracyDropTask, output_dir: str) -> None:
        self.task = task
        self.output_dir = Path(output_dir)

    def run(self):
        logger.info(f"Running accuracy drop analysis: {self.task.name}")

        baseline_df = self._load_results(self.task.baseline_csv)
        corrupt_df = self._load_results(self.task.corrupted_csv)

        baseline_accuracy = self._calculate_accuracy_per_class(baseline_df)
        corrupt_accuracy = self._calculate_accuracy_per_class(corrupt_df)

        accuracy_drop = baseline_accuracy["accuracy"] - corrupt_accuracy["accuracy"]

        accuracy_drop_df = pd.DataFrame(
            {
                "synset": baseline_accuracy["synset"],
                "y_true": baseline_accuracy["y_true"],
                "accuracy_diff": accuracy_drop,
            }
        )

        self._save_results(accuracy_drop_df)

    def _load_results(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading results from: {path}")
        return pd.read_csv(Path("results") / path)

    def _calculate_accuracy_per_class(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["y_true", "synset"])["is_correct"]
            .agg(accuracy="mean")
            .reset_index()
        )

    def _save_results(self, results_df: pd.DataFrame):
        output_path = self.output_dir / self.task.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        filename = "accuracy_drop.json"
        path_with_file = output_path / filename

        results_dict = results_df.to_dict(orient="records")

        output = {"name": self.task.name, "classes": results_dict}

        with open(path_with_file, "w") as f:
            json.dump(output, f, indent=4)

        logger.info(f"Analysis results saved to: {output_path}")
