from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
import numpy as np

__all__ = ["FeatureExtractor"]

_EMBEDDING_LAYERS: dict[str, Callable[[nn.Module], nn.Module]] = {
    "alexnet": lambda m: m.avgpool,
    "resnet18": lambda m: m.avgpool,
    "resnet50": lambda m: m.avgpool,
    "resnet152": lambda m: m.avgpool,
    "regnet_y_16gf": lambda m: m.avgpool,
    "resnext101_64x4d": lambda m: m.avgpool,
    "wide_resnet50_2": lambda m: m.avgpool,
    "wide_resnet101_2": lambda m: m.avgpool,
    "efficientnet_b0": lambda m: m.avgpool,
    "efficientnet_b4": lambda m: m.avgpool,
    "efficientnet_v2_m": lambda m: m.avgpool,
    "densenet121": lambda m: m.features,
    "mobilenet_v3_large": lambda m: m.avgpool,
    "vit_b_16": lambda m: m.encoder,
    "vit_l_16": lambda m: m.encoder,
    "vit_h_14": lambda m: m.encoder,
    "swin_b": lambda m: m.avgpool,
    "swin_v2_b": lambda m: m.avgpool,
    "maxvit_t": lambda m: m.classifier[0],
    "convnext_base": lambda m: m.classifier[1],
    "convnext_large": lambda m: m.classifier[1],
}

_NEEDS_SPATIAL_FLATTEN = {
    "alexnet",
    "resnet18",
    "resnet50",
    "resnet152",
    "regnet_y_16gf",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "efficientnet_b0",
    "efficientnet_b4",
    "efficientnet_v2_m",
    "mobilenet_v3_large",
    "swin_b",
    "swin_v2_b",
    "convnext_base",
    "convnext_large",
}

_NEEDS_ADAPTIVE_POOL = {"densenet121", "maxvit_t"}

_NEEDS_CLS_TOKEN = {"vit_b_16", "vit_l_16", "vit_h_14"}


def get_embedding_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name not in _EMBEDDING_LAYERS:
        raise ValueError(f"No embedding layer defined for model: '{model_name}'")
    return _EMBEDDING_LAYERS[model_name](model)


class FeatureExtractor:
    def __init__(self, model: nn.Module, model_name: str):
        self._model_name = model_name.lower()
        self._layer = get_embedding_layer(model, model_name)
        self._hook_handle = None
        self._buffer: list[torch.Tensor] = []

    def __enter__(self) -> FeatureExtractor:
        self._buffer.clear()
        self._hook_handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def _hook(self, _: nn.Module, __, output: torch.Tensor):
        name = self._model_name

        if name in _NEEDS_CLS_TOKEN:
            # (B, seq_len, D) – CLS token at 0
            vec = output[:, 0, :]

        elif name in _NEEDS_ADAPTIVE_POOL:
            # (B, C, H, W) → (B, C)
            vec = torch.nn.functional.adaptive_avg_pool2d(output, 1)
            vec = vec.flatten(start_dim=1)

        elif name in _NEEDS_SPATIAL_FLATTEN:
            # (B, C, 1, 1) → (B, C)
            vec = output.flatten(start_dim=1)

        else:
            vec = output.flatten(start_dim=1)

        self._buffer.append(vec.detach().cpu())

    def get(self) -> np.ndarray:
        return torch.cat(self._buffer, dim=0).numpy().astype(np.float32)

    def clear(self):
        self._buffer.clear()
