from __future__ import annotations

import torch.nn as nn
import torchattacks

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_fgsm(model: nn.Module, epsilon: float) -> torchattacks.Attack:
    atk = torchattacks.FGSM(model, eps=epsilon)
    atk.set_normalization_used(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return atk


def get_pgd(model: nn.Module, epsilon: float, steps: int = 5) -> torchattacks.Attack:
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 4, steps=steps)
    atk.set_normalization_used(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return atk
