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

from .plot import plot_fragile_dot


TASK_NAME = "fragile_dot_v2"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Cleveland dot plot — clean vs corrupted accuracy for robust/fragile synsets",
    )
    parser.add_argument("--robust", type=str, default="", help="Comma-separated robust synset IDs")
    parser.add_argument("--fragile", type=str, required=True, help="Comma-separated fragile synset IDs")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--corruption", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=".")


def run(args: argparse.Namespace) -> None:
    robust_synsets = [s.strip() for s in args.robust.split(",") if s.strip()]
    fragile_synsets = [s.strip() for s in args.fragile.split(",") if s.strip()]

    group = next(
        (g for g, cs in IMAGENET_C_CORRUPTION_GROUPS.items() if args.corruption in cs),
        None,
    )
    if group is None:
        print(f"[{TASK_NAME}] Unknown corruption: {args.corruption}", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data_path)

    clean_df = get_per_class_accuracy(
        f"{args.model}_imagenet.csv", data_path, agg_column="acc_clean"
    )
    clean_acc = dict(zip(clean_df["synset"], clean_df["acc_clean"]))

    severity_acc: dict[int, dict[str, float]] = {}
    for sev in [1, 2, 3, 4, 5]:
        fname = f"{args.model}_imagenet_c_{group}_{args.corruption}_{sev}.csv"
        try:
            df = get_per_class_accuracy(fname, data_path)
            severity_acc[sev] = dict(zip(df["synset"], df["accuracy"]))
        except FileNotFoundError:
            print(f"[{TASK_NAME}] Missing {fname}, skipping severity {sev}", file=sys.stderr)

    if not severity_acc:
        print(f"[{TASK_NAME}] No severity data found, aborting.", file=sys.stderr)
        sys.exit(1)

    label_map = get_synset_to_label_imagenet1k()
    model_display = MODELS.get(args.model, args.model)
    title = f"{model_display}  {args.corruption.replace('_', ' ').capitalize()}"

    out = Path(args.output_dir) / "images" / "fragile" / "dot" / f"{args.model}_{args.corruption}.png"
    plot_fragile_dot(
        clean_acc=clean_acc,
        severity_acc=severity_acc,
        robust_synsets=robust_synsets,
        fragile_synsets=fragile_synsets,
        title=title,
        output_path=out,
        label_map=label_map,
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Cleveland dot plot — clean vs corrupted accuracy for robust/fragile synsets"
    )
    _parser.add_argument("--robust", type=str, default="")
    _parser.add_argument("--fragile", type=str, required=True)
    _parser.add_argument("--model", type=str, required=True)
    _parser.add_argument("--corruption", type=str, required=True)
    _parser.add_argument("--data-path", type=str, default="results")
    _parser.add_argument("--output-dir", type=str, default=".")
    run(_parser.parse_args())
