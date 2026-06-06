from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from constants import IMAGENET_C_CORRUPTION_GROUPS
from paths import paths
from representations.loader import load_aligned, resolve_pair
from representations.naming import clean_name, condition_name
from utils import get_synset_to_label_imagenet1k


def _group_for_corruption(corruption: str) -> str:
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        if corruption in corruptions:
            return group
    valid = sorted(c for cs in IMAGENET_C_CORRUPTION_GROUPS.values() for c in cs)
    raise ValueError(f"Unknown corruption '{corruption}'. Valid: {valid}")


def _coherence(delta: np.ndarray) -> float:
    """R = ||mean of unit displacements|| — same formula as directional_coherence."""
    norms = np.linalg.norm(delta, axis=1)
    nonzero = norms > 0
    if not nonzero.any():
        return 0.0
    unit = np.zeros_like(delta, dtype=np.float64)
    unit[nonzero] = delta[nonzero] / norms[nonzero, None]
    return float(np.linalg.norm(unit.mean(axis=0)))


def coherence_clean(model: str, synset: str, embeddings_dir: Path) -> float:
    """Within-class coherence on clean embeddings: displacement from class centroid."""
    npy, parquet = resolve_pair(clean_name(model), embeddings_dir)
    import pandas as pd
    features = np.load(npy, mmap_mode="r")
    meta = pd.read_parquet(parquet, columns=["synset"])
    mask = meta["synset"].to_numpy() == synset
    vecs = features[mask].astype(np.float64)
    if len(vecs) == 0:
        raise ValueError(f"Synset {synset} not found in clean embeddings")
    centroid = vecs.mean(axis=0)
    delta = vecs - centroid
    return _coherence(delta)


def coherence_corrupt(
    model: str, synset: str, corruption: str, severity: int, embeddings_dir: Path
) -> float:
    """Directional coherence of clean→corrupt displacement for the given synset."""
    group = _group_for_corruption(corruption)
    features = load_aligned(
        clean_name(model),
        condition_name(model, group, corruption, severity),
        embeddings_dir,
    )
    mask = features.synsets == synset
    if not mask.any():
        raise ValueError(f"Synset {synset} not found in embeddings")
    delta = (
        features.corrupt_features[mask].astype(np.float64)
        - features.clean_features[mask].astype(np.float64)
    )
    return _coherence(delta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directional coherence R for a single synset (clean or corrupted)"
    )
    parser.add_argument("--model", default="convnext_base")
    parser.add_argument("--synset", required=True)
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--severity", type=int, default=None)
    parser.add_argument("--embeddings-dir", default=None)
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else paths.embeddings
    label_map = get_synset_to_label_imagenet1k()
    label = label_map.get(args.synset, args.synset)

    if args.corruption is None:
        R = coherence_clean(args.model, args.synset, embeddings_dir)
        condition = "clean (displacement from centroid)"
    else:
        if args.severity is None:
            parser.error("--severity required when --corruption is given")
        R = coherence_corrupt(args.model, args.synset, args.corruption, args.severity, embeddings_dir)
        condition = f"{args.corruption} severity {args.severity}"

    print(f"{args.synset}  {label}")
    print(f"condition : {condition}")
    print(f"coherence : {R:.4f}")


if __name__ == "__main__":
    main()
