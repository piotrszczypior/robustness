from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from paths import paths

from .loader import Features, load_aligned, resolve_pair
from .naming import clean_name, condition_name, group_for_corruption

__all__ = ["compute_synset_distances", "run_synset_distance"]

logger = logging.getLogger(__name__)


def _representative(vecs: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.median(vecs, axis=0)
    return np.mean(vecs, axis=0)


def _build_reps(
    feature_matrix: np.ndarray,
    synsets_all: np.ndarray,
    target_synsets: list[str],
    method: str,
    side: str,
) -> dict[str, np.ndarray]:
    reps: dict[str, np.ndarray] = {}
    for syn in target_synsets:
        mask = synsets_all == syn
        n = int(mask.sum())
        if n == 0:
            logger.warning("Synset %s not found in %s embeddings — skipping", syn, side)
            continue
        reps[syn] = _representative(feature_matrix[mask].astype(np.float64), method)
    return reps


def _load_clean_only(
    model: str, embeddings_dir: str | None
) -> tuple[np.ndarray, np.ndarray]:
    npy_path, parquet_path = resolve_pair(clean_name(model), embeddings_dir)
    F = np.load(npy_path, mmap_mode="r")
    meta = pd.read_parquet(parquet_path, columns=["synset"])
    return np.asarray(F, dtype=np.float32), meta["synset"].to_numpy()


def _cosine_matrix(
    reps_rows: dict[str, np.ndarray],
    reps_cols: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], list[str]]:
    row_keys = list(reps_rows.keys())
    col_keys = list(reps_cols.keys())
    mat = np.zeros((len(row_keys), len(col_keys)))
    for i, rk in enumerate(row_keys):
        vr = reps_rows[rk]
        nr = np.linalg.norm(vr)
        for j, ck in enumerate(col_keys):
            vc = reps_cols[ck]
            nc = np.linalg.norm(vc)
            denom = nr * nc
            sim = float(np.dot(vr, vc) / denom) if denom > 0 else 0.0
            mat[i, j] = float(np.clip(sim, -1.0, 1.0))
    return mat, row_keys, col_keys


def _load_class_index() -> dict[str, str]:
    try:
        with open(paths.imagenet_class_index) as f:
            raw = json.load(f)
        return {v[0]: v[1] for v in raw.values()}
    except (FileNotFoundError, KeyError, AttributeError):
        return {}


def _short_label(syn: str, class_index: dict[str, str]) -> str:
    name = class_index.get(syn, syn)
    return name.replace("_", " ").capitalize()[:22]


def _annotate(ax: plt.Axes, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)


def _plot_heatmaps(
    clean_reps: dict[str, np.ndarray],
    corr_reps: dict[str, np.ndarray] | None,
    model: str,
    corruption: str | None,
    severity: int | None,
    out_path: Path,
) -> None:
    class_index = _load_class_index()

    mat_cc, row_syns, col_syns_cc = _cosine_matrix(clean_reps, clean_reps)
    row_labels = [_short_label(s, class_index) for s in row_syns]
    col_labels_cc = [_short_label(s, class_index) for s in col_syns_cc]

    n_rows = len(row_syns)

    if corr_reps is None:
        # Single panel — clean × clean only
        fig, ax = plt.subplots(figsize=(max(5, n_rows * 1.1 + 1), max(4, n_rows * 0.85 + 2)))
        norm = Normalize(vmin=mat_cc.min(), vmax=mat_cc.max())
        ax.imshow(mat_cc, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(col_labels_cc)))
        ax.set_xticklabels(col_labels_cc, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title("clean vs clean", fontsize=10)
        _annotate(ax, mat_cc)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn"), ax=ax, label="cosine similarity", shrink=0.8)
    else:
        mat_cx, _, col_syns_cx = _cosine_matrix(clean_reps, corr_reps)
        col_labels_cx = [_short_label(s, class_index) for s in col_syns_cx]
        vmin = min(mat_cc.min(), mat_cx.min())
        vmax = max(mat_cc.max(), mat_cx.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_rows * 0.85 + 2)))
        corr_title = f"clean vs {corruption} sev {severity}"
        for ax, mat, col_labels, title in [
            (axes[0], mat_cc, col_labels_cc, "clean vs clean"),
            (axes[1], mat_cx, col_labels_cx, corr_title),
        ]:
            ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_title(title, fontsize=10)
            _annotate(ax, mat)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn")
        fig.colorbar(sm, ax=axes.tolist(), label="cosine similarity", shrink=0.6)

    fig.suptitle(f"{model} — cosine similarity", fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


def _distances_from_reps(
    clean_reps: dict[str, np.ndarray],
    corr_reps: dict[str, np.ndarray],
    synsets_all: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for sc, rep_c in clean_reps.items():
        norm_c = np.linalg.norm(rep_c)
        n_c = int((synsets_all == sc).sum())
        for sr, rep_r in corr_reps.items():
            n_r = int((synsets_all == sr).sum())
            diff = rep_r - rep_c
            l2 = float(np.linalg.norm(diff))
            relative_shift = float(l2 / norm_c) if norm_c > 0 else 0.0
            norm_r = np.linalg.norm(rep_r)
            denom = norm_c * norm_r
            cos_sim = float(np.clip(np.dot(rep_c, rep_r) / denom, -1.0, 1.0)) if denom > 0 else 1.0
            rows.append({
                "synset_clean": sc,
                "synset_corr": sr,
                "n_clean": n_c,
                "n_corr": n_r,
                "l2": l2,
                "cosine_dist": 1.0 - cos_sim,
                "relative_shift": relative_shift,
            })
    return pd.DataFrame(rows, columns=["synset_clean", "synset_corr", "n_clean", "n_corr", "l2", "cosine_dist", "relative_shift"])


def compute_synset_distances(
    features: Features,
    synsets_clean: list[str],
    synsets_corr: list[str],
    method: str = "centroid",
) -> pd.DataFrame:
    clean_reps = _build_reps(features.clean_features, features.synsets, synsets_clean, method, "clean")
    corr_reps = _build_reps(features.corrupt_features, features.synsets, synsets_corr, method, "corrupted")
    return _distances_from_reps(clean_reps, corr_reps, features.synsets)


def run_synset_distance(args: argparse.Namespace) -> None:
    clean_only = not args.synsets_corr

    if not clean_only and (args.corruption is None or args.severity is None):
        raise ValueError("--corruption and --severity are required when --synsets-corr is given")

    clean_stem = clean_name(args.model)

    if clean_only:
        F_clean, synsets = _load_clean_only(args.model, args.embeddings_dir)
        clean_reps = _build_reps(F_clean, synsets, args.synsets_clean, args.aggregate, "clean")
        corr_reps = None
        result = _distances_from_reps(clean_reps, clean_reps, synsets)
    else:
        group = group_for_corruption(args.corruption)
        cond_stem = condition_name(args.model, group, args.corruption, args.severity)
        logger.info("Loading embeddings: %s vs %s", clean_stem, cond_stem)
        features = load_aligned(clean_stem, cond_stem, args.embeddings_dir)
        clean_reps = _build_reps(features.clean_features, features.synsets, args.synsets_clean, args.aggregate, "clean")
        corr_reps = _build_reps(features.corrupt_features, features.synsets, args.synsets_corr, args.aggregate, "corrupted")
        result = _distances_from_reps(clean_reps, corr_reps, features.synsets)

    if result.empty:
        print("No results — check that the given synsets exist in the embeddings.")
        return

    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            result.to_csv(out_path, index=False)
        else:
            result.to_parquet(out_path, index=False)
        print(f"\nSaved: {out_path}")

    if args.heatmap:
        if args.heatmap_out:
            heatmap_path = Path(args.heatmap_out)
        else:
            stem = args.model
            if not clean_only:
                stem += f"_{args.corruption}_{args.severity}"
            heatmap_path = (
                paths.images / "representations" / "synset_distance" / f"{stem}.png"
            )
        _plot_heatmaps(clean_reps, corr_reps, args.model, args.corruption, args.severity, heatmap_path)
