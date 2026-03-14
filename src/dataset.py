from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

from config import Config

__all__ = ["get_dataset"]


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
    config: DatasetConfig, traform: Optional[transforms.Compose] = DEFAULT_TRANSFORM
) -> ImageFolderWithMetadata:
    return DatasetFactory.create(config=config, transform=traform)


class DatasetType(Enum):
    IMAGENET = "imagenet"
    IMAGENET_C = "imagenet_c"
    IMAGENET_P = "imagenet_p"
    IMAGENET_R = "imagenet_r"
    IMAGENET_A = "imagenet_a"


@dataclass(frozen=True)
class DatasetConfig:
    type: DatasetType = DatasetType.IMAGENET
    corruption: Optional[str] = None
    severity: Optional[int] = None
    perturbation: Optional[str] = None

    def get_data_path(self) -> Path:
        """Resolves the directory path based on dataset type and parameters"""
        base_path = Path(Config.DATA_ROOT)

        if not base_path.exists():
            raise FileNotFoundError(f"Directory does not exists: {base_path}")

        if self.type == DatasetType.IMAGENET_C:
            if not self.corruption or self.severity is None:
                raise ValueError("ImageNet-C requires 'corruption' and 'severity'")
            return base_path / "imagenet_c" / self.corruption / str(self.severity)

        if self.type == DatasetType.IMAGENET_P:
            if not self.perturbation:
                raise ValueError("ImageNet-P requires 'perturbation'")
            return base_path / "imagenet_p" / self.perturbation

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
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        path, target = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        filename = os.path.basename(path)
        synset = os.path.basename(os.path.dirname(path))

        metadata = {"synset": synset, "filename": filename}

        return image, target, metadata


class DatasetFactory:
    @staticmethod
    def create(
        config: DatasetConfig, transform: Optional[transforms.Compose] = None
    ) -> Dataset:
        path = config.get_data_path()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        return ImageFolderWithMetadata(
            root=str(path), transform=transform or DEFAULT_TRANSFORM
        )
