import argparse
import pandas as pd
from pathlib import Path

from task import Task
from model import MODELS
from paths import paths
from constants import IMAGENET_C_CORRUPTION_GROUPS
from utils import get_index_to_synset_and_label_imagenet1k, get_synset_to_index_imagenet1k
from .plot import prepare_synset_data, render

TASK_NAME = "mistake_dot_v2"

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
        help="Dot plot of per-class mistake distribution for a single model",
    )
    parser.add_argument("--model", type=str, required=True, help="Model key (e.g. resnet50)")
    parser.add_argument("--dataset", type=str, default="imagenet_c", choices=["imagenet_c", "imagenet_r"])
    parser.add_argument("--corruption", type=str, default=None, help="Corruption type (e.g. defocus_blur); required for imagenet_c")
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Severity level (default: 1)")
    parser.add_argument(
        "--synsets",
        type=str,
        required=True,
        help="Comma-separated synset IDs or path to file with one synset per line",
    )
    parser.add_argument("--results-dir", type=str, default=None, help="Directory with raw CSV result files")
    parser.add_argument("--output-dir", type=str, default=".", help="Base output directory (default: .)")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum prediction count to include (default: 1)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K wrong predictions per synset (default: 10)")


def _resolve_synsets(value: str) -> list[str]:
    p = Path(value)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [s.strip() for s in value.split(",") if s.strip()]


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir) if args.results_dir else paths.results

    if args.dataset == "imagenet_r":
        csv_path = results_dir / f"{args.model}_imagenet_r.csv"
        task_name = "imagenet_r"
    else:
        if not args.corruption:
            raise ValueError("--corruption is required for imagenet_c")
        group = _CORRUPTION_TO_GROUP.get(args.corruption)
        if group is None:
            raise ValueError(f"Unknown corruption '{args.corruption}'. Known: {sorted(_CORRUPTION_TO_GROUP)}")
        csv_path = results_dir / f"{args.model}_imagenet_c_{group}_{args.corruption}_{args.severity}.csv"
        task_name = f"{args.corruption}_{args.severity}"

    if not csv_path.exists():
        raise FileNotFoundError(f"Result file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    relevant = df[["synset", "y_pred"]]
    counts = relevant.groupby(["synset", "y_pred"]).size().reset_index(name="count")
    records = counts.to_dict(orient="records")

    index_to_label = get_index_to_synset_and_label_imagenet1k()
    synset_to_index = get_synset_to_index_imagenet1k()

    synsets = _resolve_synsets(args.synsets)
    model_label = MODELS.get(args.model, args.model)
    out_base = Path(args.output_dir) / "images" / "v2" / "mistake_dot"

    for synset in synsets:
        entries = prepare_synset_data(
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
            label_slug = e[1].replace(" ", "_").replace("/", "_")
            synset_label = f"{synset} ({e[1].replace('_', ' ')})"
        else:
            label_slug = "unknown"
            synset_label = synset

        out = out_base / f"{task_name}_{args.model}_{synset}_{label_slug}.png"
        render(entries, synset, synset_label, model_label, task_name, out)
        if entries:
            print(f"  -> {out}")
