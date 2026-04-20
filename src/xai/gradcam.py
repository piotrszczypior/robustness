from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dataset import DatasetConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import get_model

from .layer import get_target_layer
from .visualization import save_heatmap
from utils import resolve_device

logger = logging.getLogger("xai.gradcam")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        grad_cam = (weights * activations).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)

        grad_cam = F.interpolate(
            grad_cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        grad_cam = grad_cam.squeeze().cpu().numpy()

        min_, max_ = grad_cam.min(), grad_cam.max()
        if max_ > min_:
            grad_cam = (grad_cam - min_) / (max_ - min_)

        return grad_cam


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


def get_images(dataset: DatasetConfig, synset: str, max_images: int = 5) -> list[Path]:
    synset_dir = dataset.get_data_path() / synset
    if not synset_dir.exists():
        raise FileNotFoundError(f"Directory {synset_dir} not found")

    image_files = [
        file
        for file in synset_dir.iterdir()
        if file.suffix.lower() in (".jpg", ".jpeg")
    ]

    indices = torch.randperm(len(image_files))[:max_images]
    images = [image_files[i] for i in indices]

    logger.info(f"Processing {len(images)} images from {synset_dir}")
    return images


def run_gradcam(model_name: str, dataset_alias: str, synset: str, output_dir: str):
    device = resolve_device()

    model, transforms = get_model(model_name)
    model.to(device)
    model.eval()

    dataset = DatasetConfig.from_alias(dataset_alias)

    target_idx = get_synset_index(synset)
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    image_files = get_images(dataset, synset)

    for i, img_path in enumerate(image_files):
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transforms(img_pil).unsqueeze(0).to(device)

        heatmap = grad_cam.generate(input_tensor, target_idx)

        output = Path(output_dir) / dataset.type.value / synset
        output.mkdir(parents=True, exist_ok=True)
        output_path = output / f"gradcam_{synset}_{i}_{img_path.name}"
        save_heatmap(heatmap, img_path, output_path)
        logger.info(f"Saved: {output_path}")

    grad_cam.remove_hooks()
