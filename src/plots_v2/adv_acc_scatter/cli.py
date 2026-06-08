from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from task import Task

TASK_NAME = "adv_acc_scatter"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        TASK_NAME,
        help="Scatter: ImageNet-C accuracy (x) vs adversarial accuracy (y) per class",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True, choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon", type=float, required=True, help="e.g. 0.01568627450980392")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corruption", type=str, help="Specific corruption, e.g. shot_noise")
    group.add_argument("--group", type=str, help="Corruption group, e.g. noise (averages all)")
    parser.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--adv-dir", type=str, default="aversarial")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="images/adversarial/scatter")
    parser.add_argument("--top-n", type=int, default=10, help="Number of classes to label")


def _eps_label(eps: float) -> str:
    n = round(eps * 255)
    if abs(n / 255 - eps) < 1e-9:
        return f"{n}_255"
    return f"{eps:.8f}".rstrip("0").rstrip(".")


def _load_adv_acc(adv_dir: Path, model: str, attack: str, epsilon: float) -> pd.DataFrame:
    path = adv_dir / f"{model}_{attack}_{_eps_label(epsilon)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Adversarial CSV not found: {path}")
    df = pd.read_csv(path)
    acc = df.groupby("synset").agg(
        adv_acc=("is_correct", "mean"),
        class_name=("class_name", "first"),
    ).reset_index()
    return acc


def _load_corrupt_acc(results_dir: Path, model: str, corruption: str | None,
                      group: str | None, severity: int) -> pd.DataFrame:
    if corruption:
        pattern = f"{model}_imagenet_c_*_{corruption}_{severity}.csv"
        files = sorted(results_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No file matching {pattern} in {results_dir}")
        df = pd.read_csv(files[0])
        label = f"{corruption} sev{severity}"
    else:
        pattern = f"{model}_imagenet_c_{group}_*_{severity}.csv"
        files = sorted(results_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        label = f"{group} sev{severity}"

    acc = df.groupby("synset")["is_correct"].mean().reset_index()
    acc.columns = ["synset", "corrupt_acc"]
    return acc, label


def run(args: argparse.Namespace) -> None:
    from .plot import plot_scatter

    adv_dir = Path(args.adv_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    adv = _load_adv_acc(adv_dir, args.model, args.attack, args.epsilon)
    corrupt, corruption_label = _load_corrupt_acc(
        results_dir, args.model, args.corruption, args.group, args.severity
    )

    df = adv.merge(corrupt, on="synset", how="inner")
    if df.empty:
        print("ERROR: no common synsets after merge", file=sys.stderr)
        return

    print(f"Plotting {len(df)} classes  (adv={len(adv)}, corrupt={len(corrupt)}, common={len(df)})")

    plot_scatter(
        df=df,
        model=args.model,
        attack=args.attack,
        epsilon=args.epsilon,
        corruption_label=corruption_label,
        output_dir=output_dir,
        top_n=args.top_n,
    )
