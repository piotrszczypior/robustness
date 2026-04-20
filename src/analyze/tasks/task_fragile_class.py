from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from analyze.analyses import FragileClassTask

logger = logging.getLogger(__name__)


def run(task: FragileClassTask, output_dir: str) -> None:
    _FragileClassAnalysis(task, output_dir).run()


class _FragileClassAnalysis:
    def __init__(self, task: FragileClassTask, output_dir: str) -> None:
        self.task = task
        self.output_dir = Path(output_dir)

    def run(self):
        logger.info(f"Running analysis: '{self.task.name}'")

        baseline_df = self._load_results(self.task.baseline_csv)
        degraded_df = self._load_results(self.task.corrupted_csv)

        baseline_accuracy = self._calculate_accuracy_per_class(baseline_df)
        degraded_accuracy = self._calculate_accuracy_per_class(degraded_df)

        data = pd.merge(
            baseline_accuracy, degraded_accuracy, on=["y_true", "synset"]
        ).rename(
            columns={
                "accuracy_x": "accuracy_clean",
                "accuracy_y": "accuracy_corrupted",
            }
        )

        fragile_classes = data[
            (data["accuracy_clean"] >= 0.8) & (data["accuracy_corrupted"] <= 0.5)
        ]
        logger.info(f"Fragile classes found: {len(fragile_classes)}")

        fragile_indices = fragile_classes["y_true"].unique()
        data["is_fragile"] = np.where(data["y_true"].isin(fragile_indices), 1, 0)

        data.sort_values(by="y_true", inplace=True)

        self._save_full_results(data)
        self._save_fragile_classes(fragile_classes)

    def _load_results(self, path: str) -> pd.DataFrame:
        logger.info(f"Loading results from: {path}")
        return pd.read_csv(Path("results") / path)

    def _calculate_accuracy_per_class(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["y_true", "synset"])["is_correct"]
            .agg(accuracy="mean")
            .reset_index()
        )

    def _save_full_results(self, results_df: pd.DataFrame):
        output_path = self.output_dir / self.task.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        filename = "classes.json"
        path_with_file = output_path / filename

        results_dict = results_df.to_dict(orient="records")

        output = {"name": self.task.name, "classes": results_dict}

        with open(path_with_file, "w") as f:
            json.dump(output, f, indent=4)

        logger.info(f"Analysis results saved to: {output_path}")

    def _save_fragile_classes(self, fragile_classes_df: pd.DataFrame):
        output_path = self.output_dir / self.task.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        filename = "fragile_classes.json"
        path_with_file = output_path / filename

        output = {
            "name": self.task.name,
            "classes": fragile_classes_df.to_dict(orient="records"),
        }

        with open(path_with_file, "w") as f:
            json.dump(output, f, indent=4)
