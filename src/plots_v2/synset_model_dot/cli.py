from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import IMAGENET_C_CORRUPTION_GROUPS
from fragile.data import get_per_class_accuracy
from model import MODELS
from task import Task
from utils import get_synset_to_label_imagenet1k

from .plot import plot_synset_model_dot


TASK_NAME = "synset_model_dot_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Dot plot — clean vs corrupt accuracy per synset grouped by model",
    )
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model keys")
    parser.add_argument("--synsets", type=str, required=True, help="Comma-separated synset IDs")
    parser.add_argument("--group", type=str, default=None, help="Corruption group (blur/noise/weather/digital) — averages over all corruptions")
    parser.add_argument("--corruption", type=str, default=None, help="Specific corruption (e.g. defocus_blur) — alternative to --group")
    parser.add_argument("--severity", type=int, required=True, help="Severity level (1-5)")
    parser.add_argument("--data-path", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="images/v3/synset_model_dot")


def _load_corrupt_acc(
    model: str,
    group: str,
    corruptions: list[str],
    severity: int,
    data_path: Path,
) -> dict[str, float]:
    import pandas as pd

    frames = []
    for corruption in corruptions:
        fname = f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"
        try:
            df = get_per_class_accuracy(fname, data_path)[["synset", "accuracy"]]
            frames.append(df)
        except FileNotFoundError:
            print(f"[{TASK_NAME}] Missing {fname}, skipping", file=sys.stderr)

    if not frames:
        return {}

    agg = (
        pd.concat(frames)
        .groupby("synset")["accuracy"]
        .mean()
    )
    return agg.to_dict()


def run(args: argparse.Namespace) -> None:
    if not args.group and not args.corruption:
        print(f"[{TASK_NAME}] Provide --group or --corruption", file=sys.stderr)
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    data_path = Path(args.data_path)

    if args.corruption:
        group = next(
            (g for g, cs in IMAGENET_C_CORRUPTION_GROUPS.items() if args.corruption in cs),
            None,
        )
        if group is None:
            print(f"[{TASK_NAME}] Unknown corruption: {args.corruption}", file=sys.stderr)
            sys.exit(1)
        corruptions = [args.corruption]
        corruption_label = args.corruption.replace("_", " ")
    else:
        group = args.group
        if group not in IMAGENET_C_CORRUPTION_GROUPS:
            print(f"[{TASK_NAME}] Unknown group: {args.group}", file=sys.stderr)
            sys.exit(1)
        corruptions = IMAGENET_C_CORRUPTION_GROUPS[group]
        corruption_label = group

    clean_acc: dict[str, dict[str, float]] = {s: {} for s in synsets}
    corrupt_acc: dict[str, dict[str, float]] = {s: {} for s in synsets}

    for model in models:
        clean_df = get_per_class_accuracy(
            f"{model}_imagenet.csv", data_path, agg_column="acc_clean"
        )
        clean_map = dict(zip(clean_df["synset"], clean_df["acc_clean"]))

        corrupt_map = _load_corrupt_acc(model, group, corruptions, args.severity, data_path)

        for synset in synsets:
            if synset in clean_map:
                clean_acc[synset][model] = clean_map[synset]
            if synset in corrupt_map:
                corrupt_acc[synset][model] = corrupt_map[synset]

    label_map = get_synset_to_label_imagenet1k()
    model_labels = {m: MODELS.get(m, m) for m in models}

    title = f"{corruption_label.capitalize()}\nseverity {args.severity}"

    slug = f"{corruption_label.replace(' ', '_')}_{args.severity}_{'_'.join(models)}"
    out = Path(args.output_dir) / f"{slug}.png"

    plot_synset_model_dot(
        clean_acc=clean_acc,
        corrupt_acc=corrupt_acc,
        synsets=synsets,
        models=models,
        title=title,
        output_path=out,
        label_map=label_map,
        model_labels=model_labels,
    )
    print(f"Saved: {out}")
