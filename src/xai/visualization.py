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


def save_xai_panel(
    explanations: dict[str, np.ndarray],
    original_img_path: Path,
    output_path: Path,
) -> None:
    img = Image.open(original_img_path).convert("RGB")
    n = len(explanations) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(img)
    axes[0].axis("off")

    # titles = {
    #     "gradcam_pp": "GradCAM++",
    #     "integrated_gradients": "Integrated Gradients",
    #     "smoothgrad_ig": "SmoothGrad(IG)",
    #     "attention_rollout": "Attention Rollout",
    # }

    for i, (name, heatmap) in enumerate(explanations.items(), start=1):
        if name in ("integrated_gradients", "smoothgrad_ig"):
            heatmap_resized = (
                np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                        img.size, resample=Image.BILINEAR
                    )
                )
                / 255.0
            )

            axes[i].imshow(heatmap_resized, cmap="gray", vmin=0, vmax=1)
        else:
            heatmap_colored = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap_colored).resize(
                img.size, resample=Image.BILINEAR
            )
            overlay = Image.blend(img, heatmap_img, alpha=0.5)
            axes[i].imshow(overlay)

        axes[i].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
