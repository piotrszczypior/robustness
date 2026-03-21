from __future__ import annotations

import torchvision.models as models
import torch.nn as nn


__all__ = ["get_model"]


def get_model(name: str, pretrained: bool = True) -> nn.Module:
    return _ModelFactory.create(name, pretrained)


class _ModelFactory:
    _REGISTRY = {
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet152": models.ResNet152_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
    }

    @classmethod
    def create(cls, name: str, pretrained: bool = True) -> nn.Module:
        name = name.lower()

        if name in cls._REGISTRY:
            weights = cls._REGISTRY[name] if pretrained else None
            return models.get_model(name, weights=weights)

        raise ValueError(f"Model '{name}' not found in registry")
