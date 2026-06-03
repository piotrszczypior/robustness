from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import get_index_to_synset_and_label_imagenet1k
from constants import IMAGENET_R_SYNSETS


def main() -> None:
    index = get_index_to_synset_and_label_imagenet1k()

    result = {
        str(idx): synset_and_label
        for idx, synset_and_label in sorted(index.items())
        if synset_and_label[0] in IMAGENET_R_SYNSETS
    }

    out = Path("imagenet_r_class_index.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Written {len(result)} entries to {out}")


if __name__ == "__main__":
    main()
