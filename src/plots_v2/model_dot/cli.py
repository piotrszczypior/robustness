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

from .plot import plot_model_dot


TASK_NAME = "model_dot_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Cleveland dot plot — clean vs corrupted accuracy per synset, models as columns, corruption×severity as groups",
    )
    parser.add_argument("--synset", type=str, required=True, help="Single synset ID")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corruption", type=str, help="Comma-separated corruption names e.g. defocus_blur,glass_blur")
    group.add_argument("--group", type=str, help="Corruption group name e.g. noise, blur, weather, digital")
    parser.add_argument("--severity", type=int, nargs="+", required=True, help="Severity levels e.g. 1 2 3")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--data-path", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=".")


def run(args: argparse.Namespace) -> None:
    synset = args.synset.strip()
    if args.group:
        corruptions = IMAGENET_C_CORRUPTION_GROUPS.get(args.group)
        if not corruptions:
            print(f"[{TASK_NAME}] Unknown group '{args.group}'. Valid: {list(IMAGENET_C_CORRUPTION_GROUPS)}", file=sys.stderr)
            sys.exit(1)
    else:
        corruptions = [c.strip() for c in args.corruption.split(",") if c.strip()]
    severities = sorted(args.severity)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    data_path = Path(args.data_path)
    label_map = get_synset_to_label_imagenet1k()

    combos: list[tuple[str, int]] = [
        (corruption, severity)
        for corruption in corruptions
        for severity in severities
    ]

    clean_acc: dict[str, dict[str, float]] = {}
    for model in models:
        fname = f"{model}_imagenet.csv"
        try:
            df = get_per_class_accuracy(fname, data_path, agg_column="acc_clean")
            clean_acc[model] = dict(zip(df["synset"], df["acc_clean"]))
        except FileNotFoundError:
            print(f"[{TASK_NAME}] Missing {fname}, clean data for '{model}' unavailable", file=sys.stderr)
            clean_acc[model] = {}

    combo_acc: dict[tuple[str, int], dict[str, float]] = {}
    for corruption in corruptions:
        group = next(
            (g for g, cs in IMAGENET_C_CORRUPTION_GROUPS.items() if corruption in cs),
            None,
        )
        if group is None:
            print(f"[{TASK_NAME}] Unknown corruption '{corruption}'", file=sys.stderr)
            sys.exit(1)
        for severity in severities:
            combo_acc[(corruption, severity)] = {}
            for model in models:
                fname = f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"
                try:
                    df = get_per_class_accuracy(fname, data_path)
                    combo_acc[(corruption, severity)][model] = dict(zip(df["synset"], df["accuracy"])).get(synset, float("nan"))
                except FileNotFoundError:
                    print(f"[{TASK_NAME}] Missing {fname}, skipping", file=sys.stderr)
                    combo_acc[(corruption, severity)][model] = float("nan")

    synset_label = label_map.get(synset, synset).replace("_", " ")
    title = f"{synset_label} ({synset})"

    out = Path(args.output_dir) / "images" / "v2" / "model_dot" / f"{synset_label}.png"
    plot_model_dot(
        clean_acc=clean_acc,
        combo_acc=combo_acc,
        models=models,
        synset=synset,
        combos=combos,
        title=title,
        output_path=out,
        model_labels={m: MODELS.get(m, m) for m in models},
    )
    print(f"Saved: {out}")
