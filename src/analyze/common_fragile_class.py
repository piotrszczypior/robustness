from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze.settings import BaseAnalysisConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommonFragileClassConfig:
    name: str
    files: list[str]
    output_filename: str


def run(config: BaseAnalysisConfig, output_dir: str):
    common_fragile_config = CommonFragileClassConfig(
        name=config.name,
        files=config.content["files"],
        output_filename=config.content["output_filename"],
    )

    _CommonFragileClassAnalysis(common_fragile_config, output_dir).run()


class _CommonFragileClassAnalysis:
    def __init__(self, config: CommonFragileClassConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.index_to_synset = self._load_human_readable_labels()

    def run(self):
        logger.info(f"Running analysis: '{self.config.name}'")

        fragile_class_sets = []
        for file in self.config.files:
            fragile_classes = self._load_fragile_classes(file)
            fragile_class_sets.append(set(fragile_classes))

        if not fragile_class_sets:
            logger.warning("No fragile class files found.")
            return

        common_fragile_classes = set.intersection(*fragile_class_sets)

        logger.info(f"Common fragile classes found: {len(common_fragile_classes)}")

        selected_rows = self.index_to_synset.loc[list(common_fragile_classes)]
        print()
        print(selected_rows)
        print()

    def _load_fragile_classes(self, file: str) -> list[Any]:
        logger.info(f"Loading fragile classes from: {file}")
        path = Path("analysis/results") / file
        with open(path, "r") as f:
            data = json.load(f)

        df = pd.json_normalize(data, record_path=["classes"], meta=["name"])
        return df[df["is_fragile"] == 1]["y_true"].tolist()

    def _load_human_readable_labels(self):
        logger.info("Loading human readable labels")
        path = Path("imagenet_class_index.json")

        with open(path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict(data, orient="index", columns=["synset", "label"])
        df.index = df.index.astype(int)
        df = df.sort_index()

        return df
