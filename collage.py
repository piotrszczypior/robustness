import argparse
import random
from pathlib import Path

from PIL import Image

COLS = 6
ROWS = 2
N = COLS * ROWS

CURATED_CLASSES: list[tuple[str, str]] = [
    ("n02099601", "Golden Retriever"),
    ("n02123045", "Tabby Cat"),
    ("n02504458", "African Elephant"),
    ("n01558993", "Robin"),
    ("n01806143", "Peacock"),
    ("n02007558", "Flamingo"),
    ("n01698640", "American Alligator"),
    ("n01443537", "Goldfish"),
    ("n01882714", "Koala"),
    ("n02814533", "Beach Wagon"),
    ("n04552348", "Warplane"),
    ("n03642806", "Laptop"),
    ("n07753592", "Banana"),
    ("n07873807", "Pizza"),
    ("n09428293", "Seashore"),
    ("n02676566", "Acoustic Guitar"),
    ("n04398044", "Teapot"),
    ("n02206856", "Bee"),
]


def crop_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)


def load_class_aware(folder: Path) -> list[Image.Image]:
    result: list[Image.Image] = []
    classes = random.sample(CURATED_CLASSES, min(N, len(CURATED_CLASSES)))
    for synset_id, _ in classes:
        class_dir = folder / synset_id
        if not class_dir.is_dir():
            continue
        paths = [p for p in class_dir.iterdir() if p.suffix.lower() == ".jpeg"]
        if not paths:
            continue
        p = random.choice(paths)
        try:
            result.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not result:
        raise FileNotFoundError(
            f"No images found for curated classes in: {folder}\n"
            "Tip: use --flat if your images are not in synset subdirectories."
        )
    return result


def load_flat(folder: Path) -> list[Image.Image]:
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in ".jpeg"]
    if not paths:
        raise FileNotFoundError(f"No images found in: {folder}")
    random.shuffle(paths)
    result = []
    for p in paths[:N]:
        try:
            result.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    return result


def build_collage(
    images: list[Image.Image],
    thumb_size: int = 224,
    gap: int = 4,
) -> Image.Image:
    cols, rows = COLS, ROWS
    n = min(len(images), cols * rows)

    canvas_w = cols * thumb_size + (cols - 1) * gap
    canvas_h = rows * thumb_size + (rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for idx in range(n):
        row = idx // cols
        col = idx % cols
        x = col * (thumb_size + gap)
        y = row * (thumb_size + gap)
        thumb = crop_square(images[idx], thumb_size)
        canvas.paste(thumb, (x, y))

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a ImageNet collage")
    parser.add_argument(
        "--input_dir",
        default="data/imagenet",
        help="ImageNet val folder (synset subdirs) or flat image folder.",
    )
    parser.add_argument(
        "--output",
        default="collage_thesis.png",
        help="Output file path (default: collage_thesis.png)",
    )
    parser.add_argument(
        "--thumb_size",
        type=int,
        default=224,
        help="Tile size in pixels (default: 224 — standard ImageNet size)",
    )
    parser.add_argument(
        "--gap", type=int, default=4, help="Gap between tiles in pixels (default: 4)"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Treat input_dir as a flat folder (no synset subdirs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    folder = Path(args.input_dir)

    print(f"[1/3] Loading images from: {folder}")
    images = load_flat(folder) if args.flat else load_class_aware(folder)
    print(f"      Loaded {len(images)} images.")

    print("[2/3] Building collage…")
    collage = build_collage(images, thumb_size=args.thumb_size, gap=args.gap)

    out = Path(args.output)
    collage.save(out, dpi=(300, 300))
    print(f"[3/3] Saved: {out}  ({collage.width}×{collage.height} px)")


if __name__ == "__main__":
    main()
