from pathlib import Path
from PIL import Image, ImageDraw

# ── Konfiguracja ──────────────────────────────────────────────────────────────
CLEAN_DIR = Path("data/imagenet")
SYNSET = "n02814860"  # n02509815
THUMB_SIZE = 100
GAP = 6
BG_COLOR = (245, 245, 245)

LAYOUT = [["clean", "imgR", "imgR", "imgR", "imgR", "imgR"]]

IMG_R = [
    "data/imagenet_r/imagenet-r/n02814860/videogame_17.jpg",
    "data/imagenet_r/imagenet-r/n02814860/sticker_7.jpg",
    "data/imagenet_r/imagenet-r/n02814860/sketch_9.jpg",
    "data/imagenet_r/imagenet-r/n02814860/cartoon_14.jpg",
    "data/imagenet_r/imagenet-r/n02814860/painting_8.jpg",
]


# ── Funkcje pomocnicze ────────────────────────────────────────────────────────
def crop_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.LANCZOS)


def pick_image(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    paths = [
        p for p in folder.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")
    ]
    return paths[15] if paths else None


def load_img(path: Path, size: int) -> Image.Image:
    return crop_square(Image.open(path).convert("RGB"), size)


def placeholder(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.text((size // 2 - 25, size // 2 - 6), "missing", fill=(120, 120, 120))
    return img


# ── Główny skrypt ─────────────────────────────────────────────────────────────
def main():
    synset_dir = CLEAN_DIR / SYNSET
    src_path = pick_image(synset_dir)

    if not src_path:
        raise FileNotFoundError(f"Brak czystych zdjęć w {synset_dir}")

    print(f"Używam zdjęcia bazowego: {src_path.name}")

    # 1. Generowanie standardowych bloków dla ustalonego SEVERITY
    for i, row_items in enumerate(LAYOUT, start=1):
        n_cols = len(row_items)
        canvas_w = n_cols * THUMB_SIZE + (n_cols - 1) * GAP
        canvas = Image.new("RGB", (canvas_w, THUMB_SIZE), BG_COLOR)

        for col_idx, name in enumerate(row_items):
            x = col_idx * (THUMB_SIZE + GAP)

            if name == "clean":
                tile = load_img(src_path, THUMB_SIZE)
            else:
                path = Path(IMG_R[col_idx - 1])

                tile = (
                    load_img(path, THUMB_SIZE)
                    if path and path.exists()
                    else placeholder(THUMB_SIZE)
                )

            canvas.paste(tile, (x, 0))

        out_name = f"collage_{SYNSET}_{i}.png"
        canvas.save(out_name, dpi=(300, 300))
        print(f"Zapisano {out_name}:\n -> {' | '.join(row_items)}")

    # 2. Generowanie dodatkowego zdjęcia dla różnych poziomów severity
    # print(f"\nGeneruję pasek severity dla korupcji: '{SEVERITY_CORRUPTION}'...")

    # n_cols_sev = len(SEVERITIES) + 1  # 1 dla clean + liczba poziomów severity
    # canvas_sev_w = n_cols_sev * THUMB_SIZE + (n_cols_sev - 1) * GAP
    # canvas_sev = Image.new("RGB", (canvas_sev_w, THUMB_SIZE), BG_COLOR)

    # # Dodaj czyste zdjęcie jako pierwsze (kolumna 0)
    # clean_tile = load_img(src_path, THUMB_SIZE)
    # canvas_sev.paste(clean_tile, (0, 0))

    # # Dodaj kolejne poziomy severity (kolumny 1 do 5)
    # for idx, sev in enumerate(SEVERITIES):
    #     x = (idx + 1) * (THUMB_SIZE + GAP)
    #     folder = CORRUPT_DIR / SEVERITY_CORRUPTION / str(sev) / SYNSET
    #     path = folder / src_path.name

    #     if not path.exists():
    #         path = pick_image(folder)

    #     tile = load_img(path, THUMB_SIZE) if path and path.exists() else placeholder(THUMB_SIZE)
    #     canvas_sev.paste(tile, (x, 0))

    # out_name_sev = f"collage_{SYNSET}_severity_{SEVERITY_CORRUPTION}.png"
    # canvas_sev.save(out_name_sev, dpi=(300, 300))
    # labels = ["clean"] + [f"sev{s}" for s in SEVERITIES]
    # print(f"Zapisano {out_name_sev}:\n -> {' | '.join(labels)}")


if __name__ == "__main__":
    main()
