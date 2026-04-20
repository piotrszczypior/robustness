from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_heatmap(heatmap: np.ndarray, original_img_path: Path, output_path: Path):
    img = Image.open(original_img_path).convert("RGB")

    heatmap_colored = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored).resize(
        img.size, resample=Image.BILINEAR
    )

    overlay = Image.blend(img, heatmap_img, alpha=0.5)

    _, axes = plt.subplots(1, 2, figsize=(10, 8))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
