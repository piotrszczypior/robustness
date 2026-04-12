from __future__ import annotations

from typing import Callable, Tuple

import torchvision.models as models
import torch.nn as nn


__all__ = ["get_model"]


def get_model(name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable]:
    return _ModelFactory.create(name, pretrained)


class _ModelFactory:
    _REGISTRY = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet152": models.ResNet152_Weights.IMAGENET1K_V1,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "swin_b": models.Swin_B_Weights.IMAGENET1K_V1,
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
    }

    @classmethod
    def create(cls, name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable]:
        name = name.lower()

        if name in cls._REGISTRY:
            weights = cls._REGISTRY[name]
            model = models.get_model(name, weights=weights if pretrained else None)
            return model, weights.transforms()

        raise ValueError(f"Model '{name}' not found in registry")
