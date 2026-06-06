import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

OUT = Path("crop")
OUT.mkdir(exist_ok=True)
IMAGENET_SRC = Path("data/imagenet")

def resize_center_crop(input_path, output_path, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    img = Image.open(input_path).convert("RGB")
    tensor = transform(img)
    result = transforms.ToPILImage()(tensor)
    result.save(output_path)

def pick_image(directory: Path) -> Path:
    images = list(directory.glob("*.JPEG")) + list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
    return random.choice(images)

def main():
    with open("results/representations/attractors/dam.json", "r") as f:
        content = json.load(f)
    for i, s in enumerate(content["sources"]):
        directory = IMAGENET_SRC / s["synset"]
        image_path = pick_image(directory)
        resize_center_crop(image_path, OUT / f"{s["label"]}_{i}.png", 224)

if __name__ == "__main__":
    main()