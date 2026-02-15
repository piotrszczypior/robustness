from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from pathlib import Path

from config import Config


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_imagenet_loader():
    dataset = datasets.ImageFolder(root=str(Config.IMAGENET_ROOT), transform=TRANSFORM)

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS,
    )


def get_imagenet_c_loader(corruption="defocus_blur", severity=3):
    path = Path(Config.IMAGENET_C_ROOT) / corruption / str(severity)
    dataset = datasets.ImageFolder(root=str(path), transform=TRANSFORM)

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=Config.NUM_WORKERS,
    )


if __name__ == "__main__":
    imagenet = get_imagenet_loader()
    imagenet_c = get_imagenet_c_loader()

    print(f"ImageNet dataset size: {len(imagenet.dataset)}")
    print(f"ImageNet-C dataset size: {len(imagenet_c.dataset)}")

    print(f"ImageNet batches: {len(imagenet)}")
    print(f"ImageNet-C batches: {len(imagenet_c)}")
