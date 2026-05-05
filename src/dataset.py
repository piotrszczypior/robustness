from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from PIL import Image

from paths import paths
from utils import get_synset_to_index_imagenet1k

__all__ = ["get_dataset", "DatasetType", "DatasetConfig"]


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def get_dataset(
    config: DatasetConfig, transform: Optional[transforms.Compose] = DEFAULT_TRANSFORM
) -> ImageFolderWithMetadata:
    return _DatasetFactory.create(config=config, transform=transform)


class DatasetType(Enum):
    IMAGENET = "imagenet"
    IMAGENET_C = "imagenet_c"
    IMAGENET_P = "imagenet_p"
    IMAGENET_R = "imagenet_r"
    IMAGENET_A = "imagenet_a"
    IMAGENET_C_NATIVE = "imagenet_c_native"
    REAL_BLUR_IMAGES = "real_blur_images"


@dataclass(frozen=True)
class DatasetConfig:
    type: DatasetType = DatasetType.IMAGENET
    corruption: Optional[str] = None
    severity: Optional[int] = None
    perturbation: Optional[str] = None
    group: Optional[str] = None

    @staticmethod
    def from_alias(alias: str) -> DatasetConfig:
        """
        Creates a DatasetConfig from a string alias.
        Examples:
            - "imagenet"
            - "imagenet_c_defocus_blur_1"
        """
        parts = alias.split("_", maxsplit=3)
        try:
            dataset_type = DatasetType("_".join(parts[:3]))
            corruption, _, severity = parts[3].rpartition("_")
            return DatasetConfig(
                type=dataset_type, corruption=corruption, severity=int(severity)
            )
        except (ValueError, IndexError):
            pass

        parts = alias.split("_", maxsplit=2)
        try:
            dataset_type = DatasetType("_".join(parts[:2]))
        except ValueError:
            return DatasetConfig(type=DatasetType(parts[0]))

        if len(parts) < 3:
            return DatasetConfig(type=dataset_type)

        corruption, _, severity = parts[2].rpartition("_")
        try:
            severity = int(severity)
        except ValueError:
            return DatasetConfig(type=dataset_type)

        return DatasetConfig(
            type=dataset_type, corruption=corruption, severity=severity
        )

    def get_data_path(
        self, data_root: Optional[str] = None, with_root: bool = True
    ) -> Path:
        """Resolves the directory path based on dataset type and parameters"""
        if with_root:
            base_path = Path(data_root) if data_root else paths.data
        else:
            base_path = Path()

        if self.type == DatasetType.IMAGENET_C:
            if not self.corruption or self.severity is None:
                raise ValueError("ImageNet-C requires 'corruption' and 'severity'")
            return base_path / "imagenet_c" / self.corruption / str(self.severity)

        if self.type == DatasetType.IMAGENET_P:
            if not self.perturbation:
                raise ValueError("ImageNet-P requires 'perturbation'")
            return base_path / "imagenet_p" / self.perturbation

        if self.type == DatasetType.IMAGENET_C_NATIVE:
            return base_path / "imagenet"

        return base_path / self.type.value

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.type.value,
            "corruption": self.corruption or "none",
            "severity": self.severity or 0,
            "perturbation": self.perturbation or "none",
        }


class ImageFolderWithMetadata(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: transforms.Compose,
        dataset_config: DatasetConfig,
    ):
        super().__init__(root=root, transform=transform)
        self.config = dataset_config
        self.synset_to_index = get_synset_to_index_imagenet1k()

    def _native_corruption(self, image):
        assert self.config.corruption, "corruption must be set"
        assert self.config.severity, "severity must be set"

        img_np = np.array(image)
        # FIXME
        # img_np = corrupt(
        #     img_np,
        #     corruption_name=self.config.corruption,
        #     severity=self.config.severity,
        # )

        return Image.fromarray(img_np)

    def _remap_target_to_1k(self, synset: str) -> int:
        return self.synset_to_index[synset]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        path, target = self.samples[index]
        image = self.loader(path)

        if self.config.type == DatasetType.IMAGENET_C_NATIVE:
            image = self._native_corruption(image)

        if self.transform is not None:
            image = self.transform(image)

        filename = os.path.basename(path)
        synset = os.path.basename(os.path.dirname(path))

        target = self._remap_target_to_1k(synset)

        metadata = {"synset": synset, "filename": filename}

        return image, target, metadata


class _DatasetFactory:
    @staticmethod
    def create(
        config: DatasetConfig, transform: Optional[transforms.Compose] = None
    ) -> ImageFolderWithMetadata:
        path = config.get_data_path()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        return ImageFolderWithMetadata(
            root=str(path),
            transform=transform or DEFAULT_TRANSFORM,
            dataset_config=config,
        )
