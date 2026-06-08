#!/usr/bin/env python3
"""
Copy one representative image per corruption/severity for a given synset.

Output layout:
  output/{label}/{label}_clean.png
  output/{label}/{label}_{corruption}_{severity}.png

Examples:
  python scripts/copy_corruption_samples.py --synset n01440764 --group blur --severity 2
  python scripts/copy_corruption_samples.py --synset n01440764 --corruption contrast --severity 1 2 3 4 5 -i 4
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


CORRUPTION_GROUPS: dict[str, list[str]] = {
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
    "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
    "weather": ["brightness", "fog", "frost", "snow"],
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPEG"}


def load_label(synset: str, root: Path) -> str:
    index_path = root / "imagenet_class_index.json"
    if not index_path.exists():
        return synset
    with open(index_path) as f:
        idx = json.load(f)
    for v in idx.values():
        if v[0] == synset:
            return v[1].lower().replace(" ", "_")
    return synset


def pick_image(directory: Path, index: int = 0) -> Path | None:
    candidates = sorted(p for p in directory.iterdir() if p.suffix in IMAGE_SUFFIXES)
    if not candidates:
        return None
    return candidates[index % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy one sample image per corruption to output/{label}/"
    )
    parser.add_argument("--synset", required=True, help="ImageNet synset ID (e.g. n01440764)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--group", help="Corruption group: blur | noise | weather | digital")
    src.add_argument("--corruption", help="Single or comma-separated corruption names e.g. contrast,pixelate")
    parser.add_argument("--severity", required=True, type=int, nargs="+", help="One or more severity levels e.g. 1 2 3 4 5")
    parser.add_argument("--data-path", default="data", help="Base data directory (default: data)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("-i", "--index", type=int, default=0, help="Index of image to pick from sorted list (0-based, default: 0)")
    args = parser.parse_args()

    if args.group:
        corruptions = CORRUPTION_GROUPS.get(args.group)
        if not corruptions:
            print(f"Unknown group '{args.group}'. Valid: {list(CORRUPTION_GROUPS)}")
            sys.exit(1)
    else:
        corruptions = [c.strip() for c in args.corruption.split(",") if c.strip()]

    severities = sorted(args.severity)

    data_path = Path(args.data_path)
    root = data_path.parent
    label = load_label(args.synset, root)

    out_dir = Path(args.output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean image
    clean_dir = data_path / "imagenet" / args.synset
    ref = pick_image(clean_dir, args.index) if clean_dir.exists() else None
    if ref:
        dst = out_dir / f"{label}_clean.png"
        shutil.copy2(ref, dst)
        print(f"clean          → {dst}")
    else:
        print(f"[skip] clean directory not found: {clean_dir}")
        ref = None

    # Corrupted images — same filename as clean reference when possible
    for corruption in corruptions:
        for severity in severities:
            src_dir = data_path / "imagenet_c" / corruption / str(severity) / args.synset
            if not src_dir.exists():
                print(f"[skip] {corruption}/{severity} — directory not found")
                continue

            img = (src_dir / ref.name) if (ref and (src_dir / ref.name).exists()) else pick_image(src_dir, args.index)
            if not img:
                print(f"[skip] {corruption}/{severity} — no images found")
                continue

            dst = out_dir / f"{label}_{corruption}_{severity}.png"
            shutil.copy2(img, dst)
            print(f"{corruption}/{severity} → {dst}")


if __name__ == "__main__":
    main()
