from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def fgsm(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float,
    loss_fn: nn.Module,
) -> Tensor:
    images = images.clone().detach().requires_grad_(True)
    loss = loss_fn(model(images), labels)
    loss.backward()
    return (images + epsilon * images.grad.sign()).detach()


def pgd(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    epsilon: float,
    loss_fn: nn.Module,
    steps: int = 20,
    alpha: float | None = None,
) -> Tensor:
    if alpha is None:
        alpha = epsilon / 4

    adv = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)

    for _ in range(steps):
        adv = adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(adv), labels)
        loss.backward()
        adv = (adv + alpha * adv.grad.sign()).detach()
        # project back into epsilon-ball around original images
        adv = images + (adv - images).clamp(-epsilon, epsilon)

    return adv.detach()
