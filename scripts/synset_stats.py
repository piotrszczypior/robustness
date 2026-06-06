from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from constants import IMAGENET_C_CORRUPTION_GROUPS, IMAGENET_C_SEVERITIES
from utils import get_synset_to_label_imagenet1k

GROUPS = ["blur", "noise", "weather", "digital"]
SEVERITIES = IMAGENET_C_SEVERITIES


def synset_acc(df: pd.DataFrame, synset: str) -> float | None:
    rows = df[df["synset"] == synset]
    if rows.empty:
        return None
    return rows["is_correct"].mean()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def build_group_table(
    data_path: Path, model: str, group: str, synset: str
) -> pd.DataFrame:
    corruptions = IMAGENET_C_CORRUPTION_GROUPS[group]
    rows = []
    for corruption in corruptions:
        row = {"corruption": corruption}
        for sev in SEVERITIES:
            name = f"{model}_imagenet_c_{group}_{corruption}_{sev}.csv"
            try:
                df = load_csv(data_path / name)
                acc = synset_acc(df, synset)
                row[sev] = f"{acc:.2f}" if acc is not None else "n/a"
            except FileNotFoundError:
                row[sev] = "missing"
        rows.append(row)
    return pd.DataFrame(rows).set_index("corruption")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-corruption accuracy table for a synset and model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--synset", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("results"))
    args = parser.parse_args()

    label_map = get_synset_to_label_imagenet1k()
    label = label_map.get(args.synset, args.synset)

    print(f"\nModel: {args.model}  |  {args.synset} ({label})\n")

    # Clean ImageNet
    clean_path = args.data_path / f"{args.model}_imagenet.csv"
    try:
        clean_acc = synset_acc(load_csv(clean_path), args.synset)
        clean_str = f"{clean_acc:.2f}" if clean_acc is not None else "n/a"
    except FileNotFoundError:
        clean_str = "missing"
    print(f"Clean ImageNet accuracy: {clean_str}\n")

    # Per-group tables
    for group in GROUPS:
        print(f"=== {group.upper()} ===")
        table = build_group_table(args.data_path, args.model, group, args.synset)
        table.columns.name = "severity"
        print(table.to_string())
        print()


if __name__ == "__main__":
    main()
