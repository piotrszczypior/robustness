from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from config import Config


def get_imagenet_loader(root="image_net/"):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(root=root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=Config.NUM_WORDERS,
    )


def get_imagenet_c_loader(root="image_net_c/", corruption="gaussian_noise", severity=3):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    path = root / corruption / severity
    dataset = datasets.ImageFolder(root=path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=Config.NUM_WORDERS,
    )
