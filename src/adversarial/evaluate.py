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


def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    index_to_synset: dict,
    synset_to_label: dict,
    model_name: str,
    attack_name: str,
    epsilon: float,
    fragile_synsets: set[str],
    attack_obj=None,
    save_dir: Path | None = None,
) -> list[dict]:
    rows = []
    saved: dict[str, int] = {}

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
            rows.append(
                {
                    "model": model_name,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "synset": synset,
                    "class_name": class_name,
                    "y_true": label,
                    "y_pred": pred,
                    "is_correct": int(label == pred),
                    "is_fragile": int(synset in fragile_synsets),
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
    synset_to_label: dict,
    output_path: Path,
    output_name: str | None = None,
    fragile_synsets: set[str] = set(),
    save_images_dir: Path | None = None,
    sync_drive: bool = False,
) -> None:
    backup_dir = paths.google_colab_gdrive_path if sync_drive else None

    logger.info("Collecting baseline (clean) predictions")
    all_rows = _collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        index_to_synset=index_to_synset,
        synset_to_label=synset_to_label,
        model_name=model_name,
        attack_name="clean",
        epsilon=0.0,
        fragile_synsets=fragile_synsets,
        attack_obj=None,
        save_dir=save_images_dir,
    )

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
                synset_to_label=synset_to_label,
                model_name=model_name,
                attack_name=attack_name,
                epsilon=epsilon,
                fragile_synsets=fragile_synsets,
                attack_obj=attack_obj,
                save_dir=save_images_dir,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows).sort_values(
        ["attack", "epsilon", "synset"], ascending=True
    )
    filename = f"{output_name}.csv" if output_name else f"{model_name}.csv"
    out_file = export_results(
        df, filename, output_dir=output_path, backup_dir=backup_dir
    )
    logger.info(f"Saved {len(df)} rows to {out_file}")
