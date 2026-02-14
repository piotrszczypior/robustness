from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms

ROOT = 'data'

def get_imagenet_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    val_dataset = datasets.ImageFolder(
            ROOT,
            
            ]))