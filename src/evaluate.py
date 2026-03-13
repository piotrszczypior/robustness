from __future__ import annotations

import torch


def evaluate_per_file(model, loader, device):
    model.eval()
    model.to(device)

    with torch.inference_mode():
        for i, (inputs, targets, _) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
