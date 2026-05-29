from __future__ import annotations
import argparse
import logging
from pathlib import Path
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import get_dataset
from paths import paths
from .experiment import Experiment
from .feature_extractor import FeatureExtractor
from .writer import EmbeddingWriter
from utils import resolve_device

__all__ = ["run_embedding_evaluation"]

logger = logging.getLogger(__name__)


def run_embedding_evaluation(
    args: argparse.Namespace, model, experiments: list[Experiment], transforms
) -> None:
    backup_dir = paths.google_colab_gdrive_embeddings_path if args.sync_drive else None
    device = (
        args.device if hasattr(args, "device") and args.device else resolve_device()
    )
    logger.info(f"Using device: {device}")

    for i, experiment in enumerate(experiments):
        logger.info(f"Running experiment {i + 1}/{len(experiments)}: {experiment.name}")
        dataset_config = experiment.dataset_config
        dataset = get_dataset(config=dataset_config, transform=transforms, root=args.data_path)
        data_loader = DataLoader(
            dataset=dataset,
            num_workers=int(args.num_workers),
            batch_size=int(args.batch_size),
        )
        embeddings_path = (
            Path(args.output_path)
            / f"{args.model}_{experiment.filename_suffix}_embeddings"
        )
        run_accuracy, run_error = _extract_per_condition(
            model=model,
            data_loader=data_loader,
            device=device,
            model_name=args.model,
            embeddings_path=embeddings_path,
        )
        logger.info(
            f"Experiment {experiment.name} finished. "
            f"Accuracy: {run_accuracy:.4f}, "
            f"Error: {run_error:.4f}"
        )
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(embeddings_path.with_suffix(".npy"), backup_dir)
            shutil.copy2(embeddings_path.with_suffix(".parquet"), backup_dir)


def _extract_per_condition(
    model,
    data_loader,
    device,
    model_name: str,
    embeddings_path: Path,
) -> tuple[float, float]:
    model.eval()
    model.to(device)

    extractor = FeatureExtractor(model, model_name)
    total_correct = 0
    with EmbeddingWriter(embeddings_path) as writer, torch.inference_mode():
        for inputs, targets, batch_metadata in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = _run_batch(model, extractor, writer, inputs, targets, batch_metadata)
            total_correct += (predictions == targets).sum().item()

    accuracy = total_correct / len(data_loader.dataset)
    return accuracy, 1.0 - accuracy


def _run_batch(model, extractor, writer, inputs, targets, batch_metadata) -> torch.Tensor:
    with extractor:
        outputs = model(inputs)
        vecs = extractor.get()

    probs = F.softmax(outputs, dim=1)
    _, predictions = torch.max(probs, dim=1)

    writer.write_batch(
        filenames=batch_metadata["filename"],
        synsets=batch_metadata["synset"],
        targets=targets.cpu().numpy(),
        predictions=predictions.cpu().numpy(),
        embeddings=vecs,
    )
    return predictions
