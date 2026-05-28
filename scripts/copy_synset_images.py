#!/usr/bin/env python3
"""
Copy ImageNet-C images for a given synset to output/.

Examples:
  # single corruption, all severities
  python scripts/copy_synset_images.py --synset n01775062 --corruption defocus_blur

  # all corruptions in a group, all severities
  python scripts/copy_synset_images.py --synset n01775062 --group blur

  # clean output before copying
  python scripts/copy_synset_images.py --synset n01775062 --corruption defocus_blur --clear
"""

import argparse
import shutil
import sys
from pathlib import Path

CORRUPTION_GROUPS: dict[str, list[str]] = {
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
    "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
    "weather": ["brightness", "fog", "frost", "snow"],
}
SEVERITIES = [1, 2, 3, 4, 5]


def resolve_corruptions(args: argparse.Namespace) -> list[str]:
    if args.corruption:
        return [args.corruption]
    corruptions = CORRUPTION_GROUPS.get(args.group)
    if not corruptions:
        print(f"Unknown group '{args.group}'. Valid: {list(CORRUPTION_GROUPS)}")
        sys.exit(1)
    return corruptions


def copy_images(
    synset: str,
    corruptions: list[str],
    data_path: Path,
    output_path: Path,
) -> None:
    for corruption in corruptions:
        for severity in SEVERITIES:
            src_dir = data_path / "imagenet_c" / corruption / str(severity) / synset
            dst_dir = output_path / "imagenet_c" / corruption / str(severity) / synset

            if not src_dir.exists():
                print(f"  [skip] {src_dir} — not found")
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            images = list(src_dir.iterdir())
            for img in images:
                shutil.copy2(img, dst_dir / img.name)

            print(f"  {corruption}/sev{severity}: {len(images)} images → {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy ImageNet-C synset images to output/")
    parser.add_argument("--synset", required=True, help="ImageNet synset ID (e.g. n01775062)")
    parser.add_argument("--corruption", default=None, help="Single corruption (e.g. defocus_blur)")
    parser.add_argument("--group", default=None, help="Corruption group (blur | noise | digital | weather)")
    parser.add_argument("--imagenet", action="store_true", help="Also copy clean ImageNet images")
    parser.add_argument("--data-path", default="data", help="Base data directory (default: data)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--clear", action="store_true", help="Remove output directory before copying")
    args = parser.parse_args()

    if not args.corruption and not args.group and not args.imagenet:
        print("Provide --corruption, --group, or --imagenet.")
        sys.exit(1)

    data_path = Path(args.data_path)
    output_path = Path(args.output)

    if args.clear and output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleared {output_path}")

    print(f"Synset: {args.synset}")

    if args.imagenet:
        src_dir = data_path / "imagenet" / args.synset
        dst_dir = output_path / "imagenet" / args.synset
        if not src_dir.exists():
            print(f"  [skip] {src_dir} — not found")
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            images = list(src_dir.iterdir())
            for img in images:
                shutil.copy2(img, dst_dir / img.name)
            print(f"  imagenet: {len(images)} images → {dst_dir}")

    if args.corruption or args.group:
        corruptions = resolve_corruptions(args)
        print(f"Corruptions: {corruptions}")
        print(f"Severities: {SEVERITIES}")
        print()
        copy_images(args.synset, corruptions, data_path, output_path)


if __name__ == "__main__":
    main()
