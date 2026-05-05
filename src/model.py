from __future__ import annotations

import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
import torchvision.transforms as T
from typing import Callable, Tuple

__all__ = ["get_model"]


MODELS: dict[str, str] = {
    "resnet50": "ResNet-50",
    "resnet152": "ResNet-152",
    "efficientnet_b4": "EfficientNet-B4",
    "convnext_base": "ConvNeXt-Base",
    "vit_b_16": "ViT-B/16",
}


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
        "vit_h_14": models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
    }

    _JEPA_MODELS = {
        "vit_l_16_jepa": "facebook/ijepa_vitl16_1k",
        "vit_h_14_jepa": "facebook/ijepa_vith14_1k",
    }

    @classmethod
    def create(cls, name: str, pretrained: bool = True) -> Tuple[nn.Module, Callable]:
        name = name.lower()

        if name in cls._REGISTRY:
            weights = cls._REGISTRY[name]
            model = models.get_model(name, weights=weights if pretrained else None)
            return model, weights.transforms()

        if name in cls._JEPA_MODELS:
            return cls._get_jepa_model(name)

        raise ValueError(f"Model '{name}' not found in registry")

    @classmethod
    def _get_jepa_model(cls, name: str) -> Tuple[nn.Module, Callable]:
        hf_name = cls._JEPA_MODELS[name]
        model = AutoModel.from_pretrained(hf_name, ignore_mismatched_sizes=True)

        transforms = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return model, transforms
