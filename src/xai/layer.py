from __future__ import annotations
from typing import Callable

import torch.nn as nn


_TARGET_LAYERS: dict[str, Callable[[nn.Module], nn.Module]] = {
    "alexnet": lambda m: m.features[-1],
    "resnet18": lambda m: m.layer4[-1],
    "resnet50": lambda m: m.layer4[-1],
    "resnet152": lambda m: m.layer4[-1],
    "regnet_y_16gf": lambda m: m.trunk_output[-1],
    "resnext101_64x4d": lambda m: m.layer4[-1],
    "wide_resnet50_2": lambda m: m.layer4[-1],
    "wide_resnet101_2": lambda m: m.layer4[-1],
    "efficientnet_b0": lambda m: m.features[-1],
    "efficientnet_b4": lambda m: m.features[-1],
    "efficientnet_v2_m": lambda m: m.features[-1],
    "densenet121": lambda m: m.features[-1],
    "mobilenet_v3_large": lambda m: m.features[-1],
    "vit_b_16": lambda m: m.encoder.layers[-1].ln_1,
    "vit_l_16": lambda m: m.encoder.layers[-1].ln_1,
    "vit_h_14": lambda m: m.encoder.layers[-1].ln_1,
    "swin_b": lambda m: m.features[-1][-1].norm2,
    "swin_v2_b": lambda m: m.features[-1][-1].norm2,
    "maxvit_t": lambda m: m.blocks[-1].layers[-1],
    "convnext_base": lambda m: m.features[-1][-1].block[0],
    "convnext_large": lambda m: m.features[-4][-1].block[0],
}


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name not in _TARGET_LAYERS:
        raise ValueError(f"No target layer defined for model: {model_name}")

    return _TARGET_LAYERS[model_name](model)
