from __future__ import annotations

from constants import IMAGENET_C_CORRUPTION_GROUPS

__all__ = [
    "EMBEDDING_SUFFIX",
    "clean_name",
    "condition_name",
    "group_for_corruption",
]

EMBEDDING_SUFFIX = "embeddings"

_CLEAN_DATASET = "imagenet"


def group_for_corruption(corruption: str) -> str:
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        if corruption in corruptions:
            return group
    valid = sorted(c for cs in IMAGENET_C_CORRUPTION_GROUPS.values() for c in cs)
    raise ValueError(f"Unknown corruption '{corruption}'. Valid: {valid}")


def clean_name(model: str) -> str:
    """Embedding stem for the clean (ImageNet-Val) baseline of a model.
       
    Example: resnet50_imagenet_embeddings
    """
    return f"{model}_{_CLEAN_DATASET}_{EMBEDDING_SUFFIX}"


def condition_name(model: str, group: str, corruption: str, severity: int) -> str:
    """Embedding stem for one ImageNet-C condition.

    Example: resnet50_imagenet_c_blur_defocus_blur_1_embeddings
    """
    return f"{model}_imagenet_c_{group}_{corruption}_{severity}_{EMBEDDING_SUFFIX}"
