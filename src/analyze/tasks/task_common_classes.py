from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from analyze.analyses import CommonFragileClassTask

logger = logging.getLogger(__name__)


def run(task: CommonFragileClassTask, output_dir: str) -> None:
    _CommonFragileClassAnalysis(task, output_dir).run()


class _CommonFragileClassAnalysis:
    def __init__(self, task: CommonFragileClassTask, output_dir: str) -> None:
        self.task = task
        self.output_dir = Path(output_dir)
        self.index_to_synset = self._load_human_readable_labels()

    def run(self):
        logger.info(f"Running analysis: '{self.task.name}'")

        fragile_class_sets = [
            set(self._load_fragile_classes(file))
            for file in self.task.fragile_class_files
        ]

        if not fragile_class_sets:
            logger.warning("No fragile class files found.")
            return

        common_fragile_classes = set.intersection(*fragile_class_sets)
        logger.info(f"Common fragile classes found: {len(common_fragile_classes)}")

        selected_rows = self.index_to_synset.loc[list(common_fragile_classes)]

        self._save_results(selected_rows)
        print()
        print(selected_rows)
        print()

    def _load_fragile_classes(self, file: str) -> list[Any]:
        logger.info(f"Loading fragile classes from: {file}")
        path = Path("analysis") / file
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

    def _save_results(self, df: pd.DataFrame) -> None:
        import hashlib
        save_dir = Path("analysis/common/common_classes")
        save_dir.mkdir(parents=True, exist_ok=True)

        model_footprint = hashlib.md5("".join(self.task.models).encode()).hexdigest()[:8]

        json_path = save_dir / f"{self.task.name}_{model_footprint}.json"
        
        output_data = {
            "metadata": {
                "task_name": self.task.name,
                "models_count": len(self.task.fragile_class_files),
                "model_footprint": model_footprint,
                "input_files": self.task.fragile_class_files
            },
            "common_classes": df.reset_index().rename(columns={"index": "index"}).to_dict(orient="records")
        }

        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)

        latex_path = save_dir / f"{self.task.name}_{model_footprint}_table.tex"
        
        with open(latex_path, "w") as f:
            latex_str = df.to_latex(
                column_format="cll",
                caption=f"Common fragile classes for {self.task.name} | Models: {self.task.models}",
                label=f"tab:{self.task.name}_{model_footprint}"
            )
            f.write(latex_str)

        logger.info(f"Results saved with footprint: {model_footprint}")