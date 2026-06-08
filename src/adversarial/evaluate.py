from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from checkpoint import export_results
from paths import paths
from .attacks import get_fgsm, get_pgd

logger = logging.getLogger(__name__)

_ATTACK_FACTORIES = {"fgsm": get_fgsm, "pgd": get_pgd}


def _eps_label(eps: float) -> str:
    """Convert epsilon to a filename-safe string, e.g. 1/255 → '1_255'."""
    n = round(eps * 255)
    if abs(n / 255 - eps) < 1e-9:
        return f"{n}_255"
    return f"{eps:.8f}".rstrip("0").rstrip(".")


def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    index_to_synset: dict,
    model_name: str,
    attack_name: str,
    epsilon: float,
    attack_obj=None,
    save_dir: Path | None = None,
) -> list[dict]:
    rows = []
    saved: dict[str, int] = {}
    img_idx_per_class: dict[int, int] = {}

    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if attack_obj is not None:
            adv_images = attack_obj(images, labels)
        else:
            adv_images = images

        with torch.inference_mode():
            preds = model(adv_images).argmax(dim=1)

        for i, (label, pred) in enumerate(zip(labels.tolist(), preds.tolist())):
            synset, class_name = index_to_synset[label]
            img_idx = img_idx_per_class.get(label, 0)
            img_idx_per_class[label] = img_idx + 1
            rows.append(
                {
                    "model": model_name,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "synset": synset,
                    "img_idx": img_idx,
                    "class_name": class_name,
                    "y_true": label,
                    "y_pred": pred,
                    "is_correct": int(label == pred),
                }
            )

            if save_dir and saved.get(synset, 0) < 5:
                img_dir = save_dir / synset / attack_name
                img_dir.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(
                    adv_images[i].detach().cpu(),
                    img_dir / f"{saved.get(synset, 0):03d}.png",
                )
                saved[synset] = saved.get(synset, 0) + 1

    return rows


def run_adversarial_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    attacks: list[str],
    epsilons: list[float],
    device: torch.device,
    model_name: str,
    index_to_synset: dict,
    output_path: Path,
    output_prefix: str | None = None,
    save_images_dir: Path | None = None,
    sync_drive: bool = False,
) -> None:
    backup_dir = paths.google_colab_gdrive_adversarial_path if sync_drive else None
    prefix = output_prefix or model_name

    for attack_name in attacks:
        factory = _ATTACK_FACTORIES[attack_name]
        for epsilon in epsilons:
            logger.info(f"Running {attack_name.upper()} eps={epsilon:.6f}")
            attack_obj = factory(model, epsilon)
            rows = _collect_predictions(
                model=model,
                dataloader=dataloader,
                device=device,
                index_to_synset=index_to_synset,
                model_name=model_name,
                attack_name=attack_name,
                epsilon=epsilon,
                attack_obj=attack_obj,
                save_dir=save_images_dir,
            )

            df = pd.DataFrame(rows).sort_values(["synset", "y_true"], ascending=True)
            filename = f"{prefix}_{attack_name}_{_eps_label(epsilon)}.csv"
            out_file = export_results(
                df, filename, output_dir=output_path, backup_dir=backup_dir
            )
            logger.info(f"Saved {len(df)} rows → {out_file}")
