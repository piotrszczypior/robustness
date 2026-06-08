from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from task import Task
from model import get_model
from utils import resolve_device, get_index_to_synset_and_label_imagenet1k
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
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. vit_b_16)")
    parser.add_argument(
        "--attack",
        type=str,
        default="both",
        choices=["fgsm", "pgd", "both"],
        help="Attack type (default: both)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=DEFAULT_EPSILONS,
        help="Epsilon value(s) (default: 1/255 2/255 4/255 8/255)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Max images per class (default: 50)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to ImageNet val directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aversarial",
        help="Output directory (default: aversarial)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Filename prefix (default: <model>); files: <prefix>_<attack>_<eps>.csv",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save adversarial images to images/adversarial/synsets/",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="imagenet",
        choices=["imagenet", "imagenet-c"],
    )
    parser.add_argument("--corruption", type=str, default=None)
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--sync-drive", action="store_true")


def run(args: argparse.Namespace) -> None:
    attacks = ["fgsm", "pgd"] if args.attack == "both" else [args.attack]

    model, transforms = get_model(args.model)
    device = resolve_device()
    model = model.to(device)

    index_to_synset = get_index_to_synset_and_label_imagenet1k()

    logger.info(f"Model: {args.model}")
    logger.info(f"Attacks: {attacks}, epsilons: {args.epsilon}")

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
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    full_dataset = datasets.ImageFolder(str(data_root), transform=transforms)

    counts: dict[int, int] = {}
    subset_indices = []
    for i, (_, label) in enumerate(full_dataset.samples):
        if counts.get(label, 0) >= args.samples_per_class:
            continue
        subset_indices.append(i)
        counts[label] = counts.get(label, 0) + 1

    if not subset_indices:
        raise ValueError("No samples found in the dataset")

    dataset = Subset(full_dataset, subset_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Dataset: {len(dataset)} samples across {len(counts)} classes")

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
        output_path=Path(args.output),
        output_prefix=args.output_name,
        save_images_dir=images_dir,
        sync_drive=args.sync_drive,
    )
