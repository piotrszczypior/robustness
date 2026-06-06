from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE

from constants import IMAGENET_C_CORRUPTION_GROUPS
from model import MODELS
from paths import paths

from .loader import Features, load_aligned
from .naming import clean_name, condition_name

__all__ = ["run_pca_scatter"]

logger = logging.getLogger(__name__)

_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _group_for_corruption(corruption: str) -> str:
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        if corruption in corruptions:
            return group
    valid = sorted(c for cs in IMAGENET_C_CORRUPTION_GROUPS.values() for c in cs)
    raise ValueError(f"Unknown corruption '{corruption}'. Valid: {valid}")


def _load_class_index(path: Path) -> dict[str, str]:
    try:
        with open(path) as f:
            raw = json.load(f)
        return {v[0]: v[1] for v in raw.values()}
    except (FileNotFoundError, KeyError):
        return {}


def _filter_and_sample(
    features: Features,
    synsets: list[str],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_parts: list[np.ndarray] = []
    corrupt_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    for syn in synsets:
        mask = features.synsets == syn
        indices = np.where(mask)[0]
        if len(indices) == 0:
            logger.warning("Synset %s not found in embeddings — skipping", syn)
            continue
        k = min(len(indices), n_samples)
        chosen = rng.choice(indices, size=k, replace=False)
        clean_parts.append(features.clean_features[chosen])
        corrupt_parts.append(features.corrupt_features[chosen])
        label_parts.append(np.full(k, syn, dtype=object))

    if not clean_parts:
        raise ValueError("No requested synsets found in embeddings.")

    return (
        np.vstack(clean_parts),
        np.vstack(corrupt_parts),
        np.concatenate(label_parts),
    )


def run_pca_scatter(args: argparse.Namespace) -> None:
    group = _group_for_corruption(args.corruption)

    clean_stem = clean_name(args.model)
    cond_stem = condition_name(args.model, group, args.corruption, args.severity)

    logger.info("Loading embeddings: %s vs %s", clean_stem, cond_stem)
    features = load_aligned(clean_stem, cond_stem, args.embeddings_dir)

    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    clean_vecs, corrupt_vecs, sample_synsets = _filter_and_sample(
        features, synsets, args.n_samples, rng
    )

    pca = PCA(n_components=50)
    clean_50d = pca.fit_transform(clean_vecs)
    corrupt_50d = pca.transform(corrupt_vecs)

    n_clean = len(clean_50d)
    all_50d = np.vstack([clean_50d, corrupt_50d])
    all_2d = TSNE(n_components=2, random_state=args.seed).fit_transform(all_50d)
    clean_2d = all_2d[:n_clean]
    corrupt_2d = all_2d[n_clean:]

    class_labels = _load_class_index(paths.imagenet_class_index)

    unique_synsets = [s for s in synsets if s in np.unique(sample_synsets)]
    colors = {syn: _PALETTE[i % len(_PALETTE)] for i, syn in enumerate(unique_synsets)}

    fig, ax = plt.subplots(figsize=(9, 7))

    synset_handles = []
    for syn in unique_synsets:
        mask = sample_synsets == syn
        color = colors[syn]
        label = class_labels.get(syn, syn).replace("_", " ").capitalize()
        ax.scatter(
            clean_2d[mask, 0], clean_2d[mask, 1],
            c=color, marker="o", s=25, alpha=0.7,
        )
        ax.scatter(
            corrupt_2d[mask, 0], corrupt_2d[mask, 1],
            c=color, marker="x", s=25, alpha=0.7,
        )
        synset_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markersize=8, label=label)
        )

    condition_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markersize=8, label="clean"),
        plt.Line2D([0], [0], marker="x", color="#555555",
                   markersize=8, markeredgewidth=1.5, label=f"{args.corruption} severity {args.severity}"),
    ]

    legend1 = ax.legend(handles=synset_handles, title="Synset", loc="upper left",
                        fontsize=8, title_fontsize=8, framealpha=0.8)
    ax.add_artist(legend1)
    ax.legend(handles=condition_handles, title="Condition", loc="lower right",
              fontsize=8, title_fontsize=8, framealpha=0.8)

    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(
        f"{MODELS[args.model]} — {args.corruption.capitalize()} severity {args.severity}",
        fontsize=11,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            paths.images
            / "representations"
            / "pca"
            / f"{args.model}_{args.corruption}_{args.severity}.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

