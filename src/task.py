from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

__all__ = ["Task"]


@dataclass
class Task:
    name: str
    register_fn: Callable[[argparse._SubParsersAction], None]
    run_fn: Callable[[argparse.Namespace], None]

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        self.register_fn(subparsers)

    def run(self, args: argparse.Namespace) -> None:
        self.run_fn(args)
