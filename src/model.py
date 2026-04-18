from __future__ import annotations

from typing import Callable, Tuple

import torchvision.models as models
import torch.nn as nn


__all__ = ["get_model"]


def get_model(name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable]:
    return _ModelFactory.create(name, pretrained)


class _ModelFactory:
    _REGISTRY = {
        "alexnet": models.AlexNet_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet152": models.ResNet152_Weights.IMAGENET1K_V1,
        "regnet_y_16gf": models.RegNet_Y_16GF_Weights.IMAGENET1K_V2,
        "resnext101_64x4d": models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.IMAGENET1K_V2,
        "wide_resnet101_2": models.Wide_ResNet101_2_Weights.IMAGENET1K_V2,

        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,

        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,

        "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
        "vit_l_16": models.ViT_L_16_Weights.IMAGENET1K_V1,
        "swin_b": models.Swin_B_Weights.IMAGENET1K_V1,
        "swin_v2_b": models.Swin_V2_B_Weights.IMAGENET1K_V1,
        "maxvit_t": models.MaxVit_T_Weights.IMAGENET1K_V1,
        
        "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1,

        "vit_h_14": models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    }

    @classmethod
    def create(cls, name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable]:
        name = name.lower()

        if name in cls._REGISTRY:
            weights = cls._REGISTRY[name]
            model = models.get_model(name, weights=weights if pretrained else None)
            return model, weights.transforms()

        raise ValueError(f"Model '{name}' not found in registry")
