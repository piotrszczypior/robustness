"""
corruption_collage.py  –  ImageNet-C severity collage generator

Layout per corruption:
  clean | sev_1 | sev_2 | sev_3 | sev_4 | sev_5
One row per corruption, label above each column.

Usage example:
  python corruption_collage.py \
      --clean_dir   data/imagenet/val \
      --corrupt_dir data/imagenet-c \
      --synset      n01558993 \
      --output      collage_imagenet_c.png

Directory layout assumed for ImageNet-C:
  <corrupt_dir>/<corruption_name>/<severity>/<synset>/<image>.JPEG
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Corruption catalogue (order = row order in collage) ──────────────────────
CORRUPTIONS: list[tuple[str, str]] = [
    # (folder_name,          display_label)
    ("gaussian_noise", "Gaussian Noise"),
    ("shot_noise", "Shot Noise"),
    ("impulse_noise", "Impulse Noise"),
    ("defocus_blur", "Defocus Blur"),
    ("glass_blur", "Frosted Glass Blur"),
    ("motion_blur", "Motion Blur"),
    ("zoom_blur", "Zoom Blur"),
    ("snow", "Snow"),
    ("frost", "Frost"),
    ("fog", "Fog"),
    ("brightness", "Brightness"),
    ("contrast", "Contrast"),
    ("elastic_transform", "Elastic"),
    ("pixelate", "Pixelate"),
    ("jpeg_compression", "JPEG"),
]

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# ── Curated classes (synset → readable name) ─────────────────────────────────
CURATED_CLASSES: dict[str, str] = {
    "n02099601": "Golden Retriever",
    "n02123045": "Tabby Cat",
    "n02504458": "African Elephant",
    "n01558993": "Robin",
    "n01806143": "Peacock",
    "n02007558": "Flamingo",
    "n01698640": "American Alligator",
    "n01443537": "Goldfish",
    "n01882714": "Koala",
    "n02814533": "Beach Wagon",
    "n04552348": "Warplane",
    "n03642806": "Laptop",
    "n07753592": "Banana",
    "n07873807": "Pizza",
    "n09428293": "Seashore",
    "n02676566": "Acoustic Guitar",
    "n04398044": "Teapot",
    "n02206856": "Bee",
}

# ── Visual parameters (all tuneable via CLI) ──────────────────────────────────
DEFAULT_THUMB = 112  # px – tile size
DEFAULT_GAP = 6  # px – gap between tiles
DEFAULT_LABEL_H = 22  # px – height of label bar above each tile
DEFAULT_ROW_GAP = 18  # px – extra vertical space between corruption rows
DEFAULT_FONT_SIZE = 11  # pt – label font size
DEFAULT_BG = (255, 255, 255)  # canvas background (R,G,B)
DEFAULT_LABEL_BG = (255, 255, 255)  # label area background
DEFAULT_LABEL_FG = (255, 255, 255)  # label text colour
CLEAN_LABEL = "Clean"


# ── Helpers ───────────────────────────────────────────────────────────────────


def crop_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)


def pick_image(folder: Path) -> Path | None:
    """Return a random JPEG from *folder* (non-recursive)."""
    paths = [
        p for p in folder.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")
    ]
    return paths[i] if paths else None


def load_image(path: Path, size: int) -> Image.Image:
    return crop_square(Image.open(path).convert("RGB"), size)


def try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── Core collage builder ───────────────────────────────────────────────────────


def build_corruption_collage(
    clean_dir: Path,
    corrupt_dir: Path,
    synset: str,
    corruptions: list[tuple[str, str]] = CORRUPTIONS,
    severities: list[int] = SEVERITY_LEVELS,
    thumb: int = DEFAULT_THUMB,
    gap: int = DEFAULT_GAP,
    label_h: int = DEFAULT_LABEL_H,
    row_gap: int = DEFAULT_ROW_GAP,
    font_size: int = DEFAULT_FONT_SIZE,
    bg_color: tuple = DEFAULT_BG,
    label_bg: tuple = DEFAULT_LABEL_BG,
    label_fg: tuple = DEFAULT_LABEL_FG,
) -> Image.Image:
    """
    Build one large collage:
      cols  = 1 (clean) + len(severities)
      rows  = len(corruptions)
    Each cell has a label above and the image below.
    """
    n_cols = 1 + len(severities)
    n_rows = len(corruptions)

    cell_w = thumb
    cell_h = label_h + thumb

    canvas_w = n_cols * cell_w + (n_cols - 1) * gap
    canvas_h = n_rows * (cell_h + row_gap) - row_gap  # no trailing gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)
    font = try_font(font_size)

    # Pick one source image from the synset (same image for every cell)
    synset_clean = clean_dir / synset
    src_path = pick_image(synset_clean) if synset_clean.is_dir() else None
    if src_path is None:
        raise FileNotFoundError(f"No images found in {synset_clean}")

    for row_idx, (corruption_name, corruption_label) in enumerate(corruptions):
        y_top = row_idx * (cell_h + row_gap)

        # ── Column 0: clean image ──────────────────────────────────────────
        x = 0
        _draw_cell(
            canvas,
            draw,
            font,
            img=load_image(src_path, thumb),
            label=CLEAN_LABEL,
            x=x,
            y=y_top,
            cell_w=cell_w,
            label_h=label_h,
            label_bg=label_bg,
            label_fg=label_fg,
        )

        # ── Columns 1..N: severity levels ─────────────────────────────────
        for col_idx, sev in enumerate(severities):
            x = (col_idx + 1) * (cell_w + gap)
            sev_dir = corrupt_dir / corruption_name / str(sev) / synset
            sev_path = pick_image(sev_dir) if sev_dir.is_dir() else None

            if sev_path is None:
                # grey placeholder
                tile = Image.new("RGB", (thumb, thumb), (200, 200, 200))
                label_text = f"{corruption_label}\n(sev {sev}) – missing"
            else:
                tile = load_image(sev_path, thumb)
                label_text = (
                    f"{corruption_label} – {sev}" if col_idx == 0 else f"Sev {sev}"
                )

            # Only show corruption name on first severity column
            if col_idx == 0:
                lbl = f"{corruption_label}  sev {sev}"
            else:
                lbl = f"sev {sev}"

            _draw_cell(
                canvas,
                draw,
                font,
                img=tile,
                label=lbl,
                x=x,
                y=y_top,
                cell_w=cell_w,
                label_h=label_h,
                label_bg=label_bg,
                label_fg=label_fg,
            )

    return canvas


def _draw_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    img: Image.Image,
    label: str,
    x: int,
    y: int,
    cell_w: int,
    label_h: int,
    label_bg: tuple,
    label_fg: tuple,
) -> None:
    # Label background
    draw.rectangle([x, y, x + cell_w - 1, y + label_h - 1], fill=label_bg)

    # Centred label text
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x + (cell_w - tw) // 2
    ty = y + (label_h - th) // 2
    draw.text((tx, ty), label, fill=label_fg, font=font)

    # Image tile
    canvas.paste(img, (x, y + label_h))


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a per-corruption × severity collage (ImageNet-C style)."
    )
    parser.add_argument(
        "--clean_dir",
        default="data/imagenet",
        help="ImageNet val root (synset subdirs)",
    )
    parser.add_argument(
        "--corrupt_dir",
        default="data/imagenet-c",
        help="ImageNet-C root  (<corruption>/<severity>/<synset>/)",
    )
    parser.add_argument(
        "--synset",
        default="n01558993",
        help="Synset ID to use for the collage (default: Robin)",
    )
    parser.add_argument(
        "--output", default="collage_imagenet_c_severity.png", help="Output file path"
    )
    parser.add_argument(
        "--thumb",
        type=int,
        default=DEFAULT_THUMB,
        help=f"Tile size in px (default: {DEFAULT_THUMB})",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP,
        help=f"Gap between tiles in px (default: {DEFAULT_GAP})",
    )
    parser.add_argument(
        "--label_h",
        type=int,
        default=DEFAULT_LABEL_H,
        help=f"Label bar height in px (default: {DEFAULT_LABEL_H})",
    )
    parser.add_argument(
        "--row_gap",
        type=int,
        default=DEFAULT_ROW_GAP,
        help=f"Extra vertical gap between rows (default: {DEFAULT_ROW_GAP})",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help=f"Label font size in pt (default: {DEFAULT_FONT_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=None,
        help="Subset of corruptions by folder name (default: all 15)",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        default=None,
        help="Subset of severity levels e.g. 1 3 5 (default: 1–5)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Filter corruptions / severities if requested
    corruptions = CORRUPTIONS
    if args.corruptions:
        names = set(args.corruptions)
        corruptions = [(n, l) for n, l in CORRUPTIONS if n in names]

    severities = args.severities or SEVERITY_LEVELS

    print(f"[1/3] Building collage for synset '{args.synset}'")
    print(f"      {len(corruptions)} corruptions × {len(severities)} severity levels")

    collage = build_corruption_collage(
        clean_dir=Path(args.clean_dir),
        corrupt_dir=Path(args.corrupt_dir),
        synset=args.synset,
        corruptions=corruptions,
        severities=severities,
        thumb=args.thumb,
        gap=args.gap,
        label_h=args.label_h,
        row_gap=args.row_gap,
        font_size=args.font_size,
    )

    out = Path(args.output)
    collage.save(out, dpi=(300, 300))
    print(f"[2/3] Saved: {out}  ({collage.width}×{collage.height} px)")
    print("[3/3] Done.")


if __name__ == "__main__":
    main()
