from __future__ import annotations

import argparse
from contextlib import nullcontext
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_dataset
from evaluate.checkpoint import MetricsExporter

from config import Config
from evaluate.writer import EmbeddingWriter, ResultAccumulator
from evaluate.feature_extractor import FeatureExtractor
from paths import paths

from .experiment import Experiment

__all__ = ["run_evaluation", "evaluate_per_file"]

logger = logging.getLogger(__name__)


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation(
    args: argparse.Namespace, model, experiments: list[Experiment], transforms
):
    backup_dir = paths.google_colab_gdrive_path if args.sync_drive else None
    exporter = MetricsExporter(output_dir=args.output_path, backup_dir=backup_dir)
    device = resolve_device()
    logger.info(f"Using device: {device}")

    for i, experiment in enumerate(experiments):
        logger.info(f"Running experiment {i + 1}/{len(experiments)}: {experiment.name}")
        dataset_config = experiment.dataset_config
        dataset = get_dataset(config=dataset_config, traform=transforms)
        data_loader = DataLoader(
            dataset=dataset,
            num_workers=int(args.num_workers or Config.NUM_WORKERS),
            batch_size=int(args.batch_size or Config.BATCH_SIZE),
        )

        embeddings_path = (
            Path(args.output_path)
            / f"{args.model}_{experiment.filename_suffix}_embeddings.jsonl"
            if args.extract_features
            else None
        )

        results, run_accuracy, run_error = evaluate_per_file(
            model=model,
            data_loader=data_loader,
            device=device,
            run_metadata=dataset_config.metadata,
            model_name=args.model,
            embeddings_path=embeddings_path,
        )

        logger.info(
            f"Experiment {experiment.name} finished. "
            f"Accuracy: {run_accuracy:.4f}, "
            f"Error: {run_error:.4f}"
        )

        exporter.export(
            data_df=results.to_dataframe(),
            filename=f"{args.model}_{experiment.filename_suffix}.csv",
        )


def evaluate_per_file(
    model,
    data_loader,
    device,
    run_metadata,
    model_name,
    embeddings_path: Path | None = None,
):
    model.eval()
    model.to(device)
    results = ResultAccumulator(model_name=model_name).with_metadata(run_metadata)

    extractor = FeatureExtractor(model, model_name) if embeddings_path else None
    writer_ctx = EmbeddingWriter(embeddings_path) if embeddings_path else nullcontext()

    total_correct = 0

    with writer_ctx as writer, torch.inference_mode():
        for inputs, targets, batch_metadata in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if extractor:
                with extractor:
                    outputs = model(inputs)
                vecs = extractor.get()
            else:
                outputs = model(inputs)
                vecs = None

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

            if writer and vecs is not None:
                writer.write_batch(
                    filenames=batch_metadata["filename"],
                    synsets=batch_metadata["synset"],
                    targets=targets.cpu().numpy(),
                    predictions=predictions.cpu().numpy(),
                    embeddings=vecs,
                )

    accuracy = total_correct / len(data_loader.dataset)
    return results, accuracy, 1.0 - accuracy
