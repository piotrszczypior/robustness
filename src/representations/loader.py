from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paths import paths

__all__ = [
    "Features",
    "load_aligned",
    "resolve_pair",
    "list_conditions",
]

logger = logging.getLogger(__name__)

_EMBEDDINGS_SUFFIX = "_embeddings"


@dataclass
class Features:
    """Two row-aligned feature matrices for a clean / corrupted pair"""

    clean_features: np.ndarray      # (N, D) float32
    corrupt_features: np.ndarray    # (N, D) float32
    synsets: np.ndarray             # (N,)
    images: np.ndarray              # (N,)
    clean_name: str
    cond_name: str
    paired_by: str

    @property
    def n(self) -> int:
        return int(self.clean_features.shape[0])

    @property
    def dim(self) -> int:
        return int(self.clean_features.shape[1])


def list_conditions(embeddings_dir: Path | str | None = None) -> list[str]:
    directory = Path(embeddings_dir) if embeddings_dir is not None else paths.embeddings
    stems = []
    for npy in sorted(directory.glob("*.npy")):
        if npy.with_suffix(".parquet").exists():
            stems.append(npy.stem)
    return stems


def resolve_pair(
    name: str, embeddings_dir: Path | str | None = None
) -> tuple[Path, Path]:
    """Resolve a condition name to its `(.npy, .parquet)` paths.

    Accepts the bare stem (`resnet50_imagenet_c_blur_defocus_blur_1_embeddings`),
    a name without the `_embeddings` suffix, or a path to either file.
    """
    directory = Path(embeddings_dir) if embeddings_dir is not None else paths.embeddings

    candidate = Path(name)
    stem = candidate.name
    for suffix in (".npy", ".parquet"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem.endswith(_EMBEDDINGS_SUFFIX):
        stem = stem + _EMBEDDINGS_SUFFIX

    base = directory / stem
    npy_path = base.with_suffix(".npy")
    parquet_path = base.with_suffix(".parquet")

    if not npy_path.exists():
        raise FileNotFoundError(f"Missing features file: {npy_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {parquet_path}")

    return npy_path, parquet_path


def load_aligned(
    clean: str,
    cond: str,
    embeddings_dir: Path | str | None = None,
) -> Features:
    """Load a clean / condition pair and return row-aligned feature matrices.

    The two `.npy` files are memory-mapped and the two `.parquet` metadata
    files are read in full. We first try to pair by row index, asserting that
    the `image` columns are identical. If that fails, we fall back to an inner
    merge on the `image` column so misaligned exports still line up.
    """
    clean_npy, clean_parquet = resolve_pair(clean, embeddings_dir)
    cond_npy, cond_parquet = resolve_pair(cond, embeddings_dir)

    F_clean = np.load(clean_npy, mmap_mode="r")
    F_cond = np.load(cond_npy, mmap_mode="r")

    meta_clean = pd.read_parquet(clean_parquet, columns=["image", "synset"])
    meta_cond = pd.read_parquet(cond_parquet, columns=["image", "synset"])

    images_clean = meta_clean["image"].to_numpy()
    images_cond = meta_cond["image"].to_numpy()

    try:
        assert np.array_equal(images_clean, images_cond)
        paired_by = "index"
        clean_aligned = _as_float32(F_clean)
        cond_aligned = _as_float32(F_cond)
        synsets = meta_clean["synset"].to_numpy()
        images = images_clean
    except AssertionError:
        logger.warning(
            "image columns differ for %s vs %s; pairing by merge on 'image'",
            clean_npy.stem,
            cond_npy.stem,
        )
        paired_by = "image"
        idx_clean, idx_cond, synsets, images = _merge_on_image(meta_clean, meta_cond)
        clean_aligned = _as_float32(np.asarray(F_clean)[idx_clean])
        cond_aligned = _as_float32(np.asarray(F_cond)[idx_cond])

    if clean_aligned.shape != cond_aligned.shape:
        raise ValueError(
            f"Aligned matrices have mismatched shapes: "
            f"{clean_aligned.shape} vs {cond_aligned.shape}"
        )

    logger.info(
        "Aligned %s vs %s -> %d rows, dim=%d (paired by %s)",
        clean_npy.stem,
        cond_npy.stem,
        clean_aligned.shape[0],
        clean_aligned.shape[1],
        paired_by,
    )

    return Features(
        clean_features=clean_aligned,
        corrupt_features=cond_aligned,
        synsets=synsets,
        images=images,
        clean_name=clean_npy.stem,
        cond_name=cond_npy.stem,
        paired_by=paired_by,
    )


def _merge_on_image(
    meta_clean: pd.DataFrame, meta_cond: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = meta_clean.reset_index(names="_row_clean")[["_row_clean", "image", "synset"]]
    right = meta_cond.reset_index(names="_row_cond")[["_row_cond", "image"]]

    merged = left.merge(right, on="image", how="inner").sort_values("image")
    if merged.empty:
        raise ValueError("No overlapping images between clean and condition exports")

    idx_clean = merged["_row_clean"].to_numpy()
    idx_cond = merged["_row_cond"].to_numpy()
    synsets = merged["synset"].to_numpy()
    images = merged["image"].to_numpy()
    return idx_clean, idx_cond, synsets, images


def _as_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr
