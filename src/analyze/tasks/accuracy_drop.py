from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pandas as pd

from analyze.settings import BaseAnalysisConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSource:
    baseline: str
    corruption: str


@dataclass(frozen=True)
class FragileClassConfig:
    name: str
    data: DataSource


def run(config: BaseAnalysisConfig, output_dir: str):
    fragile_config = FragileClassConfig(
        name=config.name, data=DataSource(**config.content["data"])
    )
    _AccuracyDropGenerator(fragile_config, output_dir).run()


class _AccuracyDropGenerator:
    name = "Class degradation"

    def __init__(self, config: FragileClassConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)

    def run(self):
        logger.info(f"Running analysis: '{self.name}'")

        baseline_df = self._load_results(self.config.data.baseline)
        corrupt_df = self._load_results(self.config.data.corruption)

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
        output_path = self.output_dir / self.config.name
        output_path.mkdir(parents=True, exist_ok=True)

        filename = self.config.name + "_accuracy_drop.json"
        path_with_file = output_path / filename

        results_dict = results_df.to_dict(orient="records")

        output = {"name": self.config.name, "classes": results_dict}

        with open(path_with_file, "w") as f:
            json.dump(output, f, indent=4)

        logger.info(f"Analysis results saved to: {output_path}")
