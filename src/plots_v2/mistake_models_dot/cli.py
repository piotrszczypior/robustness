from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import IMAGENET_C_CORRUPTION_GROUPS
from model import MODELS
from paths import paths
from task import Task
from utils import get_index_to_synset_and_label_imagenet1k, get_synset_to_index_imagenet1k

from plots_v2.mistake_dot.plot import prepare_synset_data
from .plot import render_models


TASK_NAME = "mistake_models_dot_v2"

_CORRUPTION_TO_GROUP = {
    c: g
    for g, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items()
    for c in corruptions
}


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Vertical dot plot — prediction count per class, multiple models side by side",
    )
    parser.add_argument("--synset", type=str, required=True, help="Single synset ID")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    parser.add_argument("--dataset", type=str, default="imagenet_c", choices=["imagenet_c", "imagenet_r"])
    parser.add_argument("--corruption", type=str, default=None, help="Corruption type; required for imagenet_c")
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--top-k", type=int, default=5, help="Top K classes per model (default: 5)")
    parser.add_argument("--max-classes", type=int, default=10, help="Max unique classes on chart (default: 10)")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")


def _csv_path(results_dir: Path, model: str, dataset: str, corruption: str | None, severity: int) -> Path:
    if dataset == "imagenet_r":
        return results_dir / f"{model}_imagenet_r.csv"
    group = _CORRUPTION_TO_GROUP.get(corruption or "")
    if group is None:
        print(f"[{TASK_NAME}] Unknown corruption '{corruption}'", file=sys.stderr)
        sys.exit(1)
    return results_dir / f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"


def run(args: argparse.Namespace) -> None:
    if args.dataset == "imagenet_c" and not args.corruption:
        print(f"[{TASK_NAME}] --corruption is required for imagenet_c", file=sys.stderr)
        sys.exit(1)

    synset = args.synset.strip()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results_dir = Path(args.results_dir) if args.results_dir else paths.results

    index_to_label = get_index_to_synset_and_label_imagenet1k()
    synset_to_index = get_synset_to_index_imagenet1k()

    model_entries: dict[str, list[dict]] = {}
    for model in models:
        csv_path = _csv_path(results_dir, model, args.dataset, args.corruption, args.severity)
        if not csv_path.exists():
            print(f"[{TASK_NAME}] Missing {csv_path}, skipping model '{model}'", file=sys.stderr)
            model_entries[model] = []
            continue
        df = pd.read_csv(csv_path)[["synset", "y_pred"]]
        counts = df.groupby(["synset", "y_pred"]).size().reset_index(name="count")
        records = counts.to_dict(orient="records")
        model_entries[model] = prepare_synset_data(
            records=records,
            synset=synset,
            index_to_label=index_to_label,
            synset_to_index=synset_to_index,
            top_k=args.top_k,
            min_count=args.min_count,
        )

    true_idx = synset_to_index.get(synset)
    if true_idx is not None:
        e = index_to_label.get(true_idx, [synset, synset])
        synset_label = f"{e[1].replace('_', ' ')}  ·  {synset}"
        label_slug = e[1].replace(" ", "_").replace("/", "_")
    else:
        synset_label = synset
        label_slug = synset

    suffix = f"{args.corruption}_{args.severity}" if args.dataset == "imagenet_c" else "imagenet_r"
    out = Path(args.output_dir) / "images" / "v3" / "mistake_models_dot" / f"{label_slug}_{suffix}.png"

    render_models(
        model_entries=model_entries,
        synset=synset,
        synset_label=synset_label,
        models=models,
        model_labels={m: MODELS.get(m, m) for m in models},
        max_classes=args.max_classes,
        output_path=out,
    )
    print(f"Saved: {out}")
