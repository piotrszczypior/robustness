import argparse

from task import Task
from .corruption_similarity import find_similarities


TASK_NAME = "corruptions"


def get_task() -> Task:
    return Task(name=TASK_NAME, register_fn=register, run_fn=run)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(TASK_NAME, help="Corruption similarity analysis")
    parser.add_argument(
        "--data-path",
        type=str,
        default="results",
        help="Path to per-class accuracy CSV files",
    )


def run(args: argparse.Namespace) -> None:
    find_similarities()
