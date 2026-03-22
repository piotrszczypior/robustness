from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset import get_dataset
from checkpoint import MetricsExporter

# FIXME
from config import Config

from .experiment import Experiment

__all__ = ["ResultAccumulator", "run_evaluation", "evaluate_per_file"]

logger = logging.getLogger(__name__)


@dataclass
class ResultAccumulator:
    model_name: str
    image: list[str] = field(default_factory=list)
    synset: list[str] = field(default_factory=list)
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        filenames: tuple,
        synsets: tuple,
        targets: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
    ):
        self.image.extend(filenames)
        self.synset.extend(synsets)
        self.y_true.extend(targets)
        self.y_pred.extend(predictions)
        self.confidence.extend(confidences)

    def with_metadata(self, metadata: dict[str, Any]) -> ResultAccumulator:
        self.dataset_metadata = metadata
        return self

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "model": self.model_name,
                "image": self.image,
                "synset": self.synset,
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "confidence": self.confidence,
            }
        )
        df["is_correct"] = (df["y_true"] == df["y_pred"]).astype(int)

        for key, value in self.dataset_metadata.items():
            df[key] = value

        return df


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(config, model, experiments: list[Experiment]):
    backup_dir = Config.GOOGLE_DRIVE_PATH if config.sync_drive else None
    exporter = MetricsExporter(output_dir=config.output_path, backup_dir=backup_dir)
    device = resolve_device()

    for i, experiment in enumerate(experiments):
        logger.info(f"Running experiment {i + 1}/{len(experiments)}: {experiment.name}")
        dataset_config = experiment.dataset_config
        dataset = get_dataset(config=dataset_config)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        results, run_accuracy, run_error = evaluate_per_file(
            model, data_loader, device, dataset_config.metadata, config.model_name
        )

        logger.info(
            f"Experiment {experiment.name} finished. "
            f"Accuracy: {run_accuracy:.4f}, "
            f"Error: {run_error:.4f}"
        )

        exporter.export(
            data_df=results.to_dataframe(),
            filename=f"{config.model_name}_{experiment.name}.csv",
        )


def evaluate_per_file(model, data_loader, device, run_metadata, model_name):
    model.eval()
    model.to(device)
    results = ResultAccumulator(model_name=model_name).with_metadata(run_metadata)

    total_correct = 0

    with torch.inference_mode():
        for inputs, targets, batch_metadata in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            total_correct += (predictions == targets).sum().item()

            results.update(
                filenames=batch_metadata["filename"],
                synsets=batch_metadata["synset"],
                targets=targets.cpu().numpy(),
                predictions=predictions.cpu().numpy(),
                confidences=confidences.cpu().numpy(),
            )

    accuracy = total_correct / len(data_loader.dataset)
    error = 1.0 - accuracy

    return results, accuracy, error
