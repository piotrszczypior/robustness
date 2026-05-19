from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from checkpoint import export_results
from paths import paths
from .attacks import fgsm, pgd

logger = logging.getLogger(__name__)

_ATTACK_FNS = {"fgsm": fgsm, "pgd": pgd}


def _baseline_acc_per_synset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    index_to_synset: dict,
    save_dir: Path | None = None,
) -> dict[str, dict]:
    """Returns {synset: {"correct": int, "total": int}} on clean images."""
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    collected: dict[str, list[torch.Tensor]] = defaultdict(list) if save_dir else {}

    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for i, (label, pred) in enumerate(zip(labels.tolist(), preds.tolist())):
                synset = index_to_synset[label][0]
                stats[synset]["total"] += 1
                stats[synset]["correct"] += int(label == pred)
                if save_dir and len(collected[synset]) < 5:
                    collected[synset].append(images[i].cpu())

    if save_dir and collected:
        for synset, imgs in collected.items():
            clean_dir = save_dir / synset / "clean"
            clean_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(imgs):
                torchvision.utils.save_image(img, clean_dir / f"{idx:03d}.png")

    return stats


def _adv_acc_per_synset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    index_to_synset: dict,
    attack_name: str,
    epsilon: float,
    eps_idx: int = 1,
    save_dir: Path | None = None,
) -> dict[str, dict]:
    """Returns {synset: {"correct": int, "total": int}} on adversarial images."""
    attack_fn = _ATTACK_FNS[attack_name]
    loss_fn = nn.CrossEntropyLoss()
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    collected: dict[str, list[torch.Tensor]] = defaultdict(list) if save_dir else {}

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        model.train()  # enable gradients through BN/dropout
        adv_images = attack_fn(model, images, labels, epsilon, loss_fn)

        model.eval()
        with torch.inference_mode():
            preds = model(adv_images).argmax(dim=1)

        for i, (label, pred) in enumerate(zip(labels.tolist(), preds.tolist())):
            synset = index_to_synset[label][0]
            stats[synset]["total"] += 1
            stats[synset]["correct"] += int(label == pred)
            if save_dir and len(collected[synset]) < 5:
                collected[synset].append(adv_images[i].detach().cpu())

    if save_dir and collected:
        for synset, imgs in collected.items():
            synset_dir = save_dir / synset / attack_name
            synset_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(imgs):
                torchvision.utils.save_image(img, synset_dir / f"{idx:03d}_{eps_idx}.png")
        logger.info(f"Saved adversarial images for {len(collected)} synsets to {save_dir}")

    return stats


def run_adversarial_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    attacks: list[str],
    epsilons: list[float],
    device: torch.device,
    model_name: str,
    index_to_synset: dict,
    synset_to_label: dict,
    output_path: Path,
    fragile_synsets: set[str],
    save_images_dir: Path | None = None,
    sync_drive: bool = False,
) -> None:
    backup_dir = paths.google_colab_gdrive_path if sync_drive else None

    logger.info("Computing baseline accuracy per class")
    baseline = _baseline_acc_per_synset(model, dataloader, device, index_to_synset, save_dir=save_images_dir)

    all_rows = []
    for attack_name in attacks:
        for eps_idx, epsilon in enumerate(epsilons, start=1):
            logger.info(f"Running {attack_name.upper()} eps={epsilon:.4f}")
            adv = _adv_acc_per_synset(
                model, dataloader, device, index_to_synset, attack_name, epsilon,
                eps_idx=eps_idx, save_dir=save_images_dir,
            )

            for synset, base_stats in baseline.items():
                n = base_stats["total"]
                baseline_acc = base_stats["correct"] / n if n > 0 else 0.0
                adv_correct = adv[synset]["correct"]
                adv_acc = adv_correct / n if n > 0 else 0.0
                acc_drop = baseline_acc - adv_acc
                normalized_drop = (
                    acc_drop / baseline_acc if baseline_acc > 0 else float("nan")
                )

                all_rows.append(
                    {
                        "model": model_name,
                        "attack": attack_name,
                        "epsilon": epsilon,
                        "synset": synset,
                        "class_name": synset_to_label.get(synset, ""),
                        "n_samples": n,
                        "baseline_acc": round(baseline_acc, 4),
                        "adv_acc": round(adv_acc, 4),
                        "acc_drop": round(acc_drop, 4),
                        "normalized_drop": round(normalized_drop, 4)
                        if not pd.isna(normalized_drop)
                        else float("nan"),
                        "is_fragile": 1 if synset in fragile_synsets else 0,
                    }
                )

    df = pd.DataFrame(all_rows).sort_values(
        ["attack", "epsilon", "acc_drop"], ascending=[True, True, False]
    )
    filename = f"{model_name}.csv"
    out_file = export_results(df, filename, output_dir=output_path, backup_dir=backup_dir)
    logger.info(f"Saved {len(df)} rows to {out_file}")
