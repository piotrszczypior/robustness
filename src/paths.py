from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    """single source of truth for all paths in the project"""

    def __init__(self, root: Path | str | None = None):
        if root:
            self.root = Path(root).resolve()
        else:
            self.root = Path(__file__).parent.parent.resolve()

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def images(self) -> Path:
        return Path("images")

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs / "run.log"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def experiments_file(self) -> Path:
        return self.experiments / "experiments.yaml"

    @property
    def imagenet_class_index(self) -> Path:
        return self.root / "imagenet_class_index.json"

    @property
    def google_colab_gdrive_path(self) -> Path:
        return Path("/content/drive/MyDrive/robustness/results")


paths = ProjectPaths()
