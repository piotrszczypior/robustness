from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from constants import IMAGENET_C_CORRUPTION_GROUPS
from paths import paths
from representations.aggregate import directional_coherence
from representations.loader import load_aligned
from representations.metrics import compute_per_image_metrics
from representations.naming import clean_name, condition_name
from utils import get_synset_to_label_imagenet1k


def _group_for_corruption(corruption: str) -> str:
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items():
        if corruption in corruptions:
            return group
    valid = sorted(c for cs in IMAGENET_C_CORRUPTION_GROUPS.values() for c in cs)
    raise ValueError(f"Unknown corruption '{corruption}'. Valid: {valid}")


def _compute_synset_medians(
    model: str,
    corruption: str,
    severity: int,
    synsets: list[str],
    embeddings_dir: Path | None,
) -> dict[str, dict[str, float]]:
    group = _group_for_corruption(corruption)
    clean = clean_name(model)
    cond = condition_name(model, group, corruption, severity)

    features = load_aligned(clean, cond, embeddings_dir)
    metrics = compute_per_image_metrics(features)
    table = metrics.table

    coherence = directional_coherence(metrics.delta, table["synset"].to_numpy())

    result: dict[str, dict[str, float]] = {}
    for synset in synsets:
        mask = table["synset"] == synset
        subset = table[mask]
        if subset.empty:
            result[synset] = {
                "angular_distance": float("nan"),
                "relative_shift": float("nan"),
                "coherence": float("nan"),
            }
        else:
            result[synset] = {
                "angular_distance": float(subset["angular_distance"].median()),
                "relative_shift": float(subset["relative_shift"].median()),
                "coherence": float(coherence.get(synset, float("nan"))),
            }
    return result


def _fmt(v: float) -> str:
    return "--" if np.isnan(v) else f"{v:.2f}"


def _build_table(
    synsets: list[str],
    label_map: dict[str, str],
    data: dict[int, dict[str, dict[str, float]]],
    severities: list[int],
    fragile: set[str],
) -> str:
    n_sev = len(severities)
    col_spec = "l l " + " ".join(["r"] * n_sev * 3)

    # Column indices (1-based) for cmidrule
    cd_start, cd_end = 3, 2 + n_sev
    rs_start, rs_end = cd_end + 1, cd_end + n_sev
    co_start, co_end = rs_end + 1, rs_end + n_sev

    sev_header = " & ".join(f"S{s}" for s in severities)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"    \toprule")
    lines.append(
        f"    Synset & Class"
        f" & \\multicolumn{{{n_sev}}}{{c}}{{Cosine Distance}}"
        f" & \\multicolumn{{{n_sev}}}{{c}}{{Relative Shift}}"
        f" & \\multicolumn{{{n_sev}}}{{c}}{{Coherence}} \\\\"
    )
    lines.append(
        f"    \\cmidrule(lr){{{cd_start}-{cd_end}}}"
        f"\\cmidrule(lr){{{rs_start}-{rs_end}}}"
        f"\\cmidrule(lr){{{co_start}-{co_end}}}"
    )
    lines.append(f"    & & {sev_header} & {sev_header} & {sev_header} \\\\")
    lines.append(r"    \midrule")

    fragile_synsets = [s for s in synsets if s in fragile]
    other_synsets = [s for s in synsets if s not in fragile]

    def _row(synset: str) -> str:
        label = label_map.get(synset, synset)
        if synset in fragile:
            label += r"$^{\dagger}$"
        cd = " & ".join(_fmt(data[s][synset]["angular_distance"]) for s in severities)
        rs = " & ".join(_fmt(data[s][synset]["relative_shift"]) for s in severities)
        co = " & ".join(_fmt(data[s][synset]["coherence"]) for s in severities)
        return f"    {synset} & {label:<40} & {cd} & {rs} & {co} \\\\"

    for synset in fragile_synsets:
        lines.append(_row(synset))

    if fragile_synsets and other_synsets:
        lines.append(r"    \midrule")

    for synset in other_synsets:
        lines.append(_row(synset))

    lines.append(r"    \bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LaTeX table: cosine distance + relative shift per synset under a single corruption"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--synset", required=True,
        help="Comma-separated synset IDs, e.g. n01773157,n01774384",
    )
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severity", type=int, nargs="+", required=True)
    parser.add_argument(
        "--fragile", default="",
        help="Comma-separated synsets to mark with dagger and list first",
    )
    parser.add_argument(
        "--embeddings-dir", default=None,
        help="Directory with *_embeddings.npy/.parquet pairs (default: paths.embeddings)",
    )
    parser.add_argument("--out", default=None, help="Output .tex path (default: stdout)")
    args = parser.parse_args()

    synsets = [s.strip() for s in args.synset.split(",") if s.strip()]
    severities = sorted(args.severity)
    fragile = {s.strip() for s in args.fragile.split(",") if s.strip()}
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else paths.embeddings

    label_map = get_synset_to_label_imagenet1k()

    data: dict[int, dict[str, dict[str, float]]] = {}
    for sev in severities:
        print(f"Loading severity {sev}...", file=sys.stderr)
        data[sev] = _compute_synset_medians(
            args.model, args.corruption, sev, synsets, embeddings_dir
        )

    table = _build_table(synsets, label_map, data, severities, fragile)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"Saved: {out}", file=sys.stderr)
    else:
        print(table)


if __name__ == "__main__":
    main()
