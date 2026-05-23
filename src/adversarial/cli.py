from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from task import Task
from model import get_model
from utils import (
    resolve_device,
    get_synset_to_index_imagenet1k,
    get_index_to_synset_and_label_imagenet1k,
    get_synset_to_label_imagenet1k,
)
from paths import paths
from .evaluate import run_adversarial_evaluation

logger = logging.getLogger(__name__)

TASK_NAME = "adversarial"
DEFAULT_EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME, help="Per-class adversarial robustness evaluation (FGSM / PGD)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g. resnet50)",
    )
    parser.add_argument(
        "--fragile",
        type=str,
        default=None,
        help="Comma-separated synset IDs or file path — classes marked is_fragile=1",
    )
    parser.add_argument(
        "--robust",
        type=str,
        default=None,
        help="Comma-separated synset IDs or file path — classes marked is_fragile=0",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="both",
        choices=["fgsm", "pgd", "both"],
        help="Attack type to run (default: both)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        help="Epsilon value(s) for the attack (default: 1/255 2/255 4/255 8/255)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for data loading (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ImageNet validation set (default: <project_root>/data/imagenet/val)",
    )
    parser.add_argument(
        "--sync-drive", action="store_true", help="Sync results to Google Drive"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Max images per class to evaluate (default: 50, i.e. full ImageNet val split)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aversarial",
        help="Output directory for CSV files (default: aversarial)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save adversarial images to images/adversarial/synsets/{synset}/{attack}_eps{e}.png",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="imagenet",
        choices=["imagenet", "imagenet-c"],
        help="Dataset source: clean ImageNet val or ImageNet-C (default: imagenet)",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        help="Corruption type for imagenet-c source (e.g. defocus_blur)",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Severity level for imagenet-c source (default: 1)",
    )


def _resolve_classes(value: str) -> list[str]:
    p = Path(value)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [s.strip() for s in value.split(",") if s.strip()]


def run(args: argparse.Namespace) -> None:
    fragile_synsets = set(_resolve_classes(args.fragile)) if args.fragile else set()
    robust_synsets = set(_resolve_classes(args.robust)) if args.robust else set()
    synsets = list(fragile_synsets | robust_synsets)
    if not synsets:
        raise ValueError("No classes provided — use --fragile and/or --robust")

    attacks = ["fgsm", "pgd"] if args.attack == "both" else [args.attack]

    logger.info(f"Model: {args.model}")
    logger.info(f"Fragile: {sorted(fragile_synsets)}, Robust: {sorted(robust_synsets)}")
    logger.info(f"Attacks: {attacks}, epsilons: {args.epsilon}")

    model, transforms = get_model(args.model)
    device = resolve_device()
    model = model.to(device)

    synset_to_index = get_synset_to_index_imagenet1k()
    index_to_synset = get_index_to_synset_and_label_imagenet1k()
    synset_to_label = get_synset_to_label_imagenet1k()

    unknown = [s for s in synsets if s not in synset_to_index]
    if unknown:
        raise ValueError(f"Unknown synsets: {unknown}")

    target_indices = {synset_to_index[s] for s in synsets}

    if args.source == "imagenet-c":
        if not args.corruption:
            raise ValueError("--source imagenet-c requires --corruption")
        from dataset import DatasetConfig, DatasetType
        cfg = DatasetConfig(
            type=DatasetType.IMAGENET_C,
            corruption=args.corruption,
            severity=args.severity,
        )
        data_root = cfg.get_data_path(args.data_path)
    else:
        data_root = Path(args.data_path) if args.data_path else paths.data / "imagenet"

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    full_dataset = datasets.ImageFolder(str(data_root), transform=transforms)

    counts: dict[int, int] = {}
    subset_indices = []
    for i, (_, label) in enumerate(full_dataset.samples):
        if label not in target_indices:
            continue
        if counts.get(label, 0) >= args.samples_per_class:
            continue
        subset_indices.append(i)
        counts[label] = counts.get(label, 0) + 1

    if not subset_indices:
        raise ValueError("No samples found for the specified classes in the dataset")

    dataset = Subset(full_dataset, subset_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Dataset: {len(dataset)} samples across {len(synsets)} classes")

    images_dir = (
        Path("images") / "adversarial" / "synsets" if args.save_images else None
    )

    run_adversarial_evaluation(
        model=model,
        dataloader=dataloader,
        attacks=attacks,
        epsilons=args.epsilon,
        device=device,
        model_name=args.model,
        index_to_synset=index_to_synset,
        synset_to_label=synset_to_label,
        output_path=Path(args.output),
        fragile_synsets=fragile_synsets,
        save_images_dir=images_dir,
        sync_drive=args.sync_drive,
    )
