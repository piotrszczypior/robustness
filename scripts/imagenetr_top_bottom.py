from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from utils import get_synset_to_label_imagenet1k

AVAILABLE_MODELS = [
    "alexnet", "convnext_base", "convnext_large", "densenet121",
    "efficientnet_b0", "efficientnet_b4", "efficientnet_v2_m", "maxvit_t",
    "mobilenet_v3_large", "regnet_y_16gf", "resnet18", "resnet50",
    "resnet152", "resnext101_64x4d", "swin_b", "swin_v2_b",
    "vit_b_16", "vit_h_14", "vit_l_16", "wide_resnet50_2", "wide_resnet101_2",
]


def load_per_class_accuracy(data_path: Path, model: str) -> pd.DataFrame:
    path = data_path / f"{model}_imagenet_r.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    acc = df.groupby("synset")["is_correct"].mean().rename("accuracy").reset_index()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Best and worst ImageNet-R classes for a given model"
    )
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--data-path", type=Path, default=Path("results"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--synset", type=str, default=None)
    args = parser.parse_args()

    acc = load_per_class_accuracy(args.data_path, args.model)

    label_map = get_synset_to_label_imagenet1k()
    acc["label"] = acc["synset"].map(label_map).fillna(acc["synset"])

    if args.synset:
        row = acc[acc["synset"] == args.synset]
        if row.empty:
            print(f"Synset '{args.synset}' not found in ImageNet-R for {args.model}.")
            return
        print(f"\nModel: {args.model}  |  ImageNet-R  |  synset {args.synset}\n")
        print(row[["synset", "label", "accuracy"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return

    acc = acc.sort_values("accuracy", ascending=False).reset_index(drop=True)

    n = args.top_n
    best = acc.head(n)[["synset", "label", "accuracy"]]
    worst = acc.tail(n).iloc[::-1][["synset", "label", "accuracy"]]

    print(f"\nModel: {args.model}  |  ImageNet-R  |  top/bottom {n} classes\n")

    print(f"=== BEST {n} ===")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== WORST {n} ===")
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
