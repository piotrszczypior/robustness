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
    img: Image.Image,
    output_path: Path,
) -> None:
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
        if name in ("smoothgrad_ig",):
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
            axes[i].imshow(overlay, aspect='equal')

        axes[i].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_explanations(
    explanations: dict[str, np.ndarray],
    img: Image.Image,
    output_dir: Path,
    model_name: str,
    base_stem: str,
) -> None:
    """Save each method as a separate PNG: {model}_{method}_{base_stem}.png"""
    output_dir.mkdir(parents=True, exist_ok=True)

    _GRAYSCALE = {"smoothgrad_ig", "layer_ig"}

    for method_name, heatmap in explanations.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img)
        axes[0].axis("off")

        if method_name in _GRAYSCALE:
            heatmap_resized = (
                np.array(
                    Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                        img.size, resample=Image.BILINEAR
                    )
                ) / 255.0
            )
            axes[1].imshow(heatmap_resized, cmap="gray", vmin=0, vmax=1)
        else:
            heatmap_colored = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap_colored).resize(
                img.size, resample=Image.BILINEAR
            )
            axes[1].imshow(Image.blend(img, heatmap_img, alpha=0.3))

        axes[1].axis("off")
        plt.tight_layout()
        out_path = output_dir / f"{model_name}_{method_name}_{base_stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
