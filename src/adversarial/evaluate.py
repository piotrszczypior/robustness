from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
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
) -> dict[str, dict]:
    """Returns {synset: {"correct": int, "total": int}} on clean images."""
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for label, pred in zip(labels.tolist(), preds.tolist()):
                synset = index_to_synset[label][0]
                stats[synset]["total"] += 1
                stats[synset]["correct"] += int(label == pred)

    return stats


def _adv_acc_per_synset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    index_to_synset: dict,
    attack_name: str,
    epsilon: float,
) -> dict[str, dict]:
    """Returns {synset: {"correct": int, "total": int}} on adversarial images."""
    attack_fn = _ATTACK_FNS[attack_name]
    loss_fn = nn.CrossEntropyLoss()
    stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        model.train()  # enable gradients through BN/dropout
        adv_images = attack_fn(model, images, labels, epsilon, loss_fn)

        model.eval()
        with torch.inference_mode():
            preds = model(adv_images).argmax(dim=1)

        for label, pred in zip(labels.tolist(), preds.tolist()):
            synset = index_to_synset[label][0]
            stats[synset]["total"] += 1
            stats[synset]["correct"] += int(label == pred)

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
    sync_drive: bool = False,
) -> None:
    backup_dir = paths.google_colab_gdrive_path if sync_drive else None

    logger.info("Computing baseline accuracy per class")
    baseline = _baseline_acc_per_synset(model, dataloader, device, index_to_synset)

    for attack_name in attacks:
        for epsilon in epsilons:
            logger.info(f"Running {attack_name.upper()} eps={epsilon:.4f}")
            adv = _adv_acc_per_synset(
                model, dataloader, device, index_to_synset, attack_name, epsilon
            )

            rows = []
            for synset, base_stats in baseline.items():
                n = base_stats["total"]
                baseline_acc = base_stats["correct"] / n if n > 0 else 0.0
                adv_correct = adv[synset]["correct"]
                adv_acc = adv_correct / n if n > 0 else 0.0
                acc_drop = baseline_acc - adv_acc
                normalized_drop = (
                    acc_drop / baseline_acc if baseline_acc > 0 else float("nan")
                )

                rows.append(
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
                    }
                )

            df = pd.DataFrame(rows).sort_values("acc_drop", ascending=False)
            eps_str = f"{epsilon:.6f}".rstrip("0").rstrip(".")
            filename = f"{model_name}_{attack_name}_eps{eps_str}.csv"
            out_file = export_results(
                df, filename, output_dir=output_path, backup_dir=backup_dir
            )
            logger.info(f"Saved {len(df)} class results to {out_file}")
