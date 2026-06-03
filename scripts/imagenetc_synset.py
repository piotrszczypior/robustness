from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from fragile.experiments import EXPERIMENTS
from utils import get_synset_to_label_imagenet1k

_CORRUPTION_TO_GROUP = {
    c: g for g, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items() for c in corruptions
}


def _load_csv(data_path: Path, model: str, group: str, corruption: str, severity: int) -> pd.DataFrame:
    name = f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"
    path = data_path / name
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def synset_accuracy_single(
    data_path: Path, model: str, corruption: str, severity: int, synset: str
) -> float:
    group = _CORRUPTION_TO_GROUP.get(corruption)
    if group is None:
        raise ValueError(f"Unknown corruption: '{corruption}'. Available: {list(_CORRUPTION_TO_GROUP)}")
    df = _load_csv(data_path, model, group, corruption, severity)
    rows = df[df["synset"] == synset]
    if rows.empty:
        raise ValueError(f"Synset '{synset}' not found in {corruption}_{severity} for {model}.")
    return rows["is_correct"].mean()


def synset_accuracy_experiment(
    data_path: Path, model: str, exp_name: str, synset: str
) -> pd.DataFrame:
    variations = EXPERIMENTS[exp_name]
    records = []
    for group, corruption, severity in variations.per_unique_conditions():
        try:
            df = _load_csv(data_path, model, group, corruption, severity)
        except FileNotFoundError:
            continue
        rows = df[df["synset"] == synset]
        if rows.empty:
            continue
        records.append({
            "corruption": corruption,
            "severity": severity,
            "accuracy": rows["is_correct"].mean(),
        })
    if not records:
        raise ValueError(f"Synset '{synset}' not found in experiment '{exp_name}' for {model}.")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ImageNet-C accuracy for a specific synset"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--synset", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("results"))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", choices=list(EXPERIMENTS.keys()))
    group.add_argument("--corruption")

    parser.add_argument("--severity", type=int, choices=IMAGENET_C_SEVERITIES)
    args = parser.parse_args()

    if args.corruption and args.severity is None:
        parser.error("--severity is required when --corruption is used")

    label_map = get_synset_to_label_imagenet1k()
    label = label_map.get(args.synset, args.synset)

    print(f"\nModel: {args.model}  |  synset: {args.synset} ({label})\n")

    if args.exp:
        df = synset_accuracy_experiment(args.data_path, args.model, args.exp, args.synset)
        mean_acc = df["accuracy"].mean()
        print(f"Experiment: {args.exp}  |  mean accuracy: {mean_acc:.3f}\n")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        acc = synset_accuracy_single(args.data_path, args.model, args.corruption, args.severity, args.synset)
        print(f"Corruption: {args.corruption}  |  Severity: {args.severity}  |  Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
