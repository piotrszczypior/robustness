from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from analyze.analyses import CommonFragileMistakesTask

logger = logging.getLogger(__name__)


def run(task: CommonFragileMistakesTask, output_dir: str) -> None:
    _CommonFragileClassMistakesAnalysis(task, output_dir).run()


class _CommonFragileClassMistakesAnalysis:
    def __init__(self, task: CommonFragileMistakesTask, output_dir: str) -> None:
        self.task = task
        self.output_dir = Path(output_dir)

    def run(self):
        logger.info(f"Running analysis: '{self.task.name}'")

        common_fragile_classes = self._load_fragile_classes(self.task.common_classes)
        fragile_synsets = set(common_fragile_classes["synset"])
        model_predictions: dict[str, pd.DataFrame] = {
            model: self._load_class_prediction_file(filename, fragile_synsets)
            for model, filename in self.task.per_file_predictions.items()
        }

        results = []
        for model, df in model_predictions.items():
            grouped = df.groupby("synset")["y_pred"].apply(list).reset_index()
            grouped["y_pred_count"] = grouped["y_pred"].apply(len)
            grouped = grouped.merge(common_fragile_classes, on="synset")
            grouped["model"] = model
            grouped = grouped[["model", "synset", "y_pred", "y_pred_count"]]
            results.append(grouped)

        results_df = pd.concat(results, ignore_index=True)
        print(results_df)

        common_preds = (
            results_df.groupby("synset")["y_pred"]
            .apply(lambda x: list(set.intersection(*[set(preds) for preds in x])))
            .reset_index()
            .rename(columns={"y_pred": "common_y_pred"})
        )
        common_preds["common_y_pred_count"] = common_preds["common_y_pred"].apply(len)
        print(common_preds)

        weighted_preds = self._compute_pred_weights(results_df)
        self._save_weighted_preds(weighted_preds)

    def _load_class_prediction_file(
        self, file: str | Path, synsets: set[str]
    ) -> pd.DataFrame:
        path = Path("results") / file
        logger.info(f"Loading image predictions from: {file}")
        df = pd.read_csv(path)

        return df[df["synset"].isin(synsets)]

    def _load_fragile_classes(self, file: str) -> list[Any]:
        logger.info(f"Loading fragile classes from: {file}")
        path = Path("analysis/common/common_classes") / file
        with open(path, "r") as f:
            data = json.load(f)

        df = pd.json_normalize(data, record_path=["common_classes"], meta=["metadata"])
        return df

    def _save_weighted_preds(self, weighted_preds: pd.DataFrame) -> None:
        import hashlib

        save_dir = Path("analysis/common/common_mistakes")
        save_dir.mkdir(parents=True, exist_ok=True)

        models = list(self.task.per_file_predictions.keys())
        model_footprint = hashlib.md5("".join(models).encode()).hexdigest()[:8]

        output_data = {
            "metadata": {
                "task_name": self.task.name,
                "models": models,
                "model_footprint": model_footprint,
            },
            "weighted_preds": weighted_preds.to_dict(orient="records"),
        }

        save_path = (
            save_dir / f"{self.task.name}_weighted_mistakes_{model_footprint}.json"
        )
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)

        logger.info(f"Weighted preds saved with footprint: {model_footprint}")

    def _compute_pred_weights(self, results_df: pd.DataFrame) -> pd.DataFrame:
        from collections import Counter

        rows = []
        for (model, synset), group in results_df.groupby(["model", "synset"]):
            counter = Counter(pred for preds in group["y_pred"] for pred in preds)
            for y_pred, count in counter.items():
                rows.append(
                    {"model": model, "synset": synset, "y_pred": y_pred, "count": count}
                )

        return pd.DataFrame(rows)
