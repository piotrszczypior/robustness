from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from dataset import DatasetConfig, DatasetType
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import get_model

from .layer import get_target_layer
from .visualization import save_heatmap, save_xai_panel, save_individual_explanations
from utils import resolve_device
from paths import paths
from .methods import get_all_explanations
from utils import get_synset_to_label_imagenet1k


logger = logging.getLogger("xai.gradcam")


def load_class_index() -> Dict[str, int]:
    path = "imagenet_class_index.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r") as f:
        class_idx = json.load(f)

    return {v[0]: int(k) for k, v in class_idx.items()}


def get_synset_index(synset: str) -> int:
    synset_to_idx = load_class_index()

    if synset not in synset_to_idx:
        raise ValueError(f"Synset {synset} not found in ImageNet class index")

    return synset_to_idx[synset]


_MATCHABLE_TYPES = {DatasetType.IMAGENET, DatasetType.IMAGENET_C}


def _list_images(synset_dir: Path) -> list[Path]:
    return [f for f in synset_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")]


def get_images(dataset: DatasetConfig, synset: str, max_images: int = 5, data_root: Optional[str] = None) -> list[Path]:
    synset_dir = dataset.get_data_path(data_root) / synset
    if not synset_dir.exists():
        raise FileNotFoundError(f"Directory {synset_dir} not found")

    image_files = _list_images(synset_dir)
    indices = torch.randperm(len(image_files))[:max_images]
    images = [image_files[i] for i in indices]

    logger.info(f"Processing {len(images)} images from {synset_dir}")
    return images


def get_matched_images(
    datasets: list[tuple[str, DatasetConfig]],
    synset: str,
    max_images: int = 5,
    sample_range: Optional[tuple[int, int]] = None,
    data_root: Optional[str] = None,
) -> list[tuple[str, int, Path]]:
    """Returns (dataset_alias, sample_index, image_path) triples.
    ImageNet + ImageNet-C variants share the same selected indices (same source images).
    ImageNet-R / ImageNet-A get independently sampled images.
    Index is the position in the sorted file list of the synset directory.

    When sample_range=(start, end) is given, indices [start, end) from the sorted
    file list are used instead of random sampling.
    """
    matchable = [
        (alias, cfg) for alias, cfg in datasets if cfg.type in _MATCHABLE_TYPES
    ]
    independent = [
        (alias, cfg) for alias, cfg in datasets if cfg.type not in _MATCHABLE_TYPES
    ]

    result: list[tuple[str, int, Path]] = []

    if matchable:
        primary_alias, primary = matchable[0]
        primary_dir = primary.get_data_path(data_root) / synset
        if not primary_dir.exists():
            raise FileNotFoundError(f"Directory {primary_dir} not found")
        all_files = sorted(_list_images(primary_dir))
        if sample_range is not None:
            start, end = sample_range
            selected_indices = list(range(start, min(end, len(all_files))))
        else:
            selected_indices = torch.randperm(len(all_files))[:max_images].tolist()
        logger.info(
            f"Selected indices {selected_indices} from {primary_alias}/{synset}"
        )

        for alias, cfg in matchable:
            for idx in selected_indices:
                result.append(
                    (alias, idx, cfg.get_data_path(data_root) / synset / all_files[idx].name)
                )

    for alias, cfg in independent:
        synset_dir = cfg.get_data_path(data_root) / synset
        if not synset_dir.exists():
            raise FileNotFoundError(f"Directory {synset_dir} not found")
        all_files = sorted(_list_images(synset_dir))
        if sample_range is not None:
            start, end = sample_range
            selected_indices = list(range(start, min(end, len(all_files))))
        else:
            selected_indices = torch.randperm(len(all_files))[:max_images].tolist()
        for idx in selected_indices:
            result.append((alias, idx, synset_dir / all_files[idx].name))

    return result


def run_xai(
    model_name: str,
    dataset_aliases: list[str],
    synset: str,
    output_dir: str,
    sample_range: Optional[tuple[int, int]] = None,
    data_root: Optional[str] = None,
    layer_ig: bool = False,
    sync_drive: bool = False,
    save_individual: bool = False,
):
    device = resolve_device()

    model, transforms = get_model(model_name)
    model.to(device)
    model.eval()
    synset_to_label = get_synset_to_label_imagenet1k()

    datasets = [(alias, DatasetConfig.from_alias(alias)) for alias in dataset_aliases]

    target_idx = get_synset_index(synset)
    target_layer = get_target_layer(model, model_name)

    samples = get_matched_images(datasets, synset, sample_range=sample_range, data_root=data_root)

    for dataset_alias, sample_idx, img_path in samples:
        dataset_label = dataset_alias.replace("/", "_")
        label = synset_to_label[synset]
        if sample_range is not None:
            base_stem = f"{dataset_label}_{synset}_{label}_{sample_idx}"
        else:
            base_stem = f"{dataset_label}_{synset}_{label}"
        out_dir = Path(output_dir) / model_name / synset
        output_path = out_dir / f"{model_name}_{base_stem}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transforms(img_pil).unsqueeze(0).to(device)
        explanations = get_all_explanations(
            model, model_name, target_layer, input_tensor, target_idx, layer_ig=layer_ig
        )

        save_xai_panel(explanations, img_path, output_path)

        if save_individual:
            save_individual_explanations(
                explanations, img_path, out_dir, model_name, base_stem
            )

        if sync_drive:
            drive_path = paths.google_colab_gdrive_xai_path / output_path.relative_to(output_dir)
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, drive_path)
