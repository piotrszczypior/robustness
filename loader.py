from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torchvision.datasets as datasets

from src.config import Config

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


class DatasetType(Enum):
    IMAGENET = "imagenet"
    IMAGENET_C = "imagenet_c"


@dataclass(frozen=True)
class DatasetConfig:
    type: DatasetType = DatasetType.IMAGENET
    corruption: Optional[str] = None
    severity: Optional[int] = None
    synset: Optional[str] = None

    def should_load_full_dataset(self) -> bool:
        return self.synset == None

    def get_data_path(self) -> Path:
        data_root = self._get_data_path()

        if data_root.exists():
            return data_root

        raise FileNotFoundError(f"Directory does not exists: {data_root}")

    def _get_data_path(self) -> Path:
        base_path = Path(Config.DATA_ROOT)

        if self.type == DatasetType.IMAGENET:
            return base_path / "imagenet"

        if self.type == DatasetType.IMAGENET_C:
            if not self.corruption or self.severity is None:
                raise ValueError(
                    "Corruption and severity must be specified for ImageNet-C"
                )
            return base_path / "imagenet_c" / self.corruption / str(self.severity)

        raise ValueError(f"Unknown dataset type: {self.type}")


class ImageNetDataModule:
    @staticmethod
    def get_dataset(config: DatasetConfig) -> Dataset:
        path = config.get_data_path()
        dataset = datasets.ImageFolder(root=path, transform=TRANSFORM)

        if config.should_load_full_dataset():
            return dataset

        target_idx = dataset.class_to_idx[config.synset]
        indices = [i for i, label in enumerate(dataset.targets) if label == target_idx]

        return Subset(dataset, indices)


if __name__ == "__main__":
    imagenet = ImageNetDataModule.get_dataset(
        DatasetConfig(type=DatasetType.IMAGENET, synset="n01695060")
    )
    for img, i in imagenet:
        print(i)
    print(f"Full ImageNet Val size: {len(imagenet)}")
