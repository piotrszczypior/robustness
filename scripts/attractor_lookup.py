#!/usr/bin/env python3
"""
Find which ImageNet-C settings drive the most images from other classes into a given
attractor synset, and which source classes are most often misclassified as it.

Examples:
  ./run.sh scripts/attractor_lookup.py --attractor n01608432 --model resnet50
  ./run.sh scripts/attractor_lookup.py --attractor n01608432 --model resnet50 --severities 1 2 3 --group blur
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES


def load_label_maps() -> tuple[dict[int, str], dict[str, str], dict[str, int]]:
    with open(ROOT / "imagenet_class_index.json") as f:
        index = json.load(f)
    idx_to_synset = {int(i): v[0] for i, v in index.items()}
    synset_to_label = {v[0]: v[1].replace("_", " ") for v in index.values()}
    synset_to_idx = {v[0]: int(i) for i, v in index.items()}
    return idx_to_synset, synset_to_label, synset_to_idx


def cond_pred_path(results_dir: Path, model: str, group: str, corruption: str, severity: int) -> Path:
    return results_dir / f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Attractor lookup for a given synset and model")
    parser.add_argument("--attractor", required=True, help="Synset of the attractor class (e.g. n01608432)")
    parser.add_argument("--model", required=True, help="Model key (e.g. resnet50)")
    parser.add_argument("--results-dir", default=str(ROOT / "results"), help="Directory with prediction CSVs")
    parser.add_argument("--severities", type=int, nargs="+", default=IMAGENET_C_SEVERITIES)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--group", help="Corruption group: blur | noise | weather | digital")
    src.add_argument("--corruption", help="Single corruption name")
    parser.add_argument("--top-k", type=int, default=15, help="Rows to show in each table (default: 15)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    _, synset_to_label, synset_to_idx = load_label_maps()

    attractor_synset = args.attractor
    attractor_idx = synset_to_idx.get(attractor_synset)
    if attractor_idx is None:
        print(f"[error] Unknown synset: {attractor_synset}", file=sys.stderr)
        sys.exit(1)
    attractor_label = synset_to_label.get(attractor_synset, attractor_synset)

    # Build scope: (group, corruption) pairs
    scope: list[tuple[str, str]] = []
    if args.corruption:
        group = next(
            (g for g, cs in IMAGENET_C_CORRUPTION_GROUPS.items() if args.corruption in cs),
            None,
        )
        if group is None:
            print(f"[error] Unknown corruption: {args.corruption}", file=sys.stderr)
            sys.exit(1)
        scope = [(group, args.corruption)]
    elif args.group:
        corruptions = IMAGENET_C_CORRUPTION_GROUPS.get(args.group)
        if not corruptions:
            print(f"[error] Unknown group: {args.group}", file=sys.stderr)
            sys.exit(1)
        scope = [(args.group, c) for c in corruptions]
    else:
        scope = [
            (group, corruption)
            for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items()
            for corruption in corruptions
        ]

    print(f"Attractor: {attractor_synset} ({attractor_label}) | model: {args.model}")
    print(f"Severities: {sorted(args.severities)} | settings: {len(scope) * len(args.severities)}\n")

    records: list[dict] = []
    source_totals: Counter = Counter()

    for group, corruption in scope:
        for severity in sorted(args.severities):
            path = cond_pred_path(results_dir, args.model, group, corruption, severity)
            if not path.exists():
                continue

            df = pd.read_csv(path, usecols=["synset", "y_pred"])
            non_attractor = df[df["synset"] != attractor_synset]
            attracted = non_attractor[non_attractor["y_pred"] == attractor_idx]

            n = len(attracted)
            total = len(non_attractor)
            frac = n / total if total else 0.0

            sources: Counter = Counter(attracted["synset"].tolist())
            source_totals.update(sources)

            records.append({
                "corruption": corruption,
                "severity": severity,
                "n_attracted": n,
                "frac": frac,
                "sources": sources,
            })

    if not records:
        print("No data found.")
        return

    records.sort(key=lambda r: -r["n_attracted"])

    # Table 1: top settings
    col_w = max(len(r["corruption"]) for r in records) + 2
    header = f"{'corruption':<{col_w}} {'sev':>3}  {'n_attracted':>11}  {'frac%':>6}"
    print("Top settings by attraction")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in records[: args.top_k]:
        print(
            f"{r['corruption']:<{col_w}} {r['severity']:>3}  "
            f"{r['n_attracted']:>11}  {r['frac'] * 100:>5.2f}%"
        )

    print()

    # Table 2: top source classes
    top_sources = source_totals.most_common(args.top_k)
    # For each source, find the setting with highest count for it
    def top_setting_for(synset: str) -> str:
        best = max(records, key=lambda r: r["sources"].get(synset, 0))
        return f"{best['corruption']}_{best['severity']}"

    label_w = max((len(synset_to_label.get(s, s)) for s, _ in top_sources), default=10) + 2
    print("Top source classes attracted to", f"{attractor_synset} ({attractor_label})")
    print("-" * 80)
    hdr2 = f"{'synset':<12} {'label':<{label_w}} {'total':>7}  top_setting"
    print(hdr2)
    print("-" * 80)
    for synset, total in top_sources:
        label = synset_to_label.get(synset, synset)
        ts = top_setting_for(synset)
        print(f"{synset:<12} {label:<{label_w}} {total:>7}  {ts}")


if __name__ == "__main__":
    main()
