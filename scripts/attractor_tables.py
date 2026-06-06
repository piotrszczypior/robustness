from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import IMAGENET_C_SEVERITIES

PRED_DIR = ROOT / "results"
ATTR_DIR = ROOT / "results" / "representations" / "attractors"
OUT_DIR = ATTR_DIR / "tables"

SEVERITIES = IMAGENET_C_SEVERITIES


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_label_map() -> dict[str, str]:
    with open(ROOT / "imagenet_class_index.json") as f:
        index = json.load(f)
    return {syn: label for syn, label in index.values()}


def synset_acc(df: pd.DataFrame, synset: str) -> float | None:
    rows = df[df["synset"] == synset]
    if rows.empty:
        return None
    return float(rows["is_correct"].mean())


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def synset_row(synset: str, model: str, group: str, corruption: str) -> dict:
    """One row: clean accuracy + s1..s5 for a given synset/model/corruption."""
    clean_df = load_csv(PRED_DIR / f"{model}_imagenet.csv")
    row: dict[str, str] = {"clean": fmt(synset_acc(clean_df, synset) if clean_df is not None else None)}
    for sev in SEVERITIES:
        df = load_csv(PRED_DIR / f"{model}_imagenet_c_{group}_{corruption}_{sev}.csv")
        row[f"s{sev}"] = fmt(synset_acc(df, synset) if df is not None else None)
    return row


def build_attractor_table(synset: str, models: list[str], group: str, corruption: str) -> pd.DataFrame:
    """Rows = models, cols = clean + s1..s5."""
    rows = [{"model": m, **synset_row(synset, m, group, corruption)} for m in models]
    return pd.DataFrame(rows).set_index("model")


def build_sources_table(
    sources: list[dict],
    models: list[str],
    group: str,
    corruption: str,
) -> pd.DataFrame:
    """Rows = sources, cols = clean + s1..s5 (mean over the source's models).

    Each source has its own models list (intersection with the attractor's models).
    Falls back to the full attractor models list if source models are missing.
    """
    col_keys = ["clean"] + [f"s{s}" for s in SEVERITIES]
    rows = []
    for src in sources:
        synset = src["synset"]
        label = src.get("label", synset)
        src_models = src.get("models") or models

        per_model = [synset_row(synset, m, group, corruption) for m in src_models]
        stats: dict[str, str] = {}
        for key in col_keys:
            vals = [float(r[key]) for r in per_model if r[key] != "n/a"]
            if not vals:
                stats[f"{key}_mean"] = "n/a"
                stats[f"{key}_std"] = "n/a"
            else:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                stats[f"{key}_mean"] = fmt(mean)
                stats[f"{key}_std"] = fmt(std)

        rows.append({"source": label, **stats})
    return pd.DataFrame(rows).set_index("source")


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    col_fmt = "l" + "r" * len(df.columns)
    header = "& " + " & ".join(df.columns) + r" \\"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for idx, row in df.iterrows():
        lines.append(str(idx) + " & " + " & ".join(row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_pair(out_dir: Path, stem: str, df: pd.DataFrame, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text(df.to_string() + "\n")
    (out_dir / f"{stem}.tex").write_text(to_latex(df, caption, label))


def main() -> None:
    attr_json = ATTR_DIR / "attractors.json"
    if not attr_json.exists():
        sys.exit(f"Not found: {attr_json}\nRun scripts/attractors.py first.")

    with open(attr_json) as f:
        records: list[dict] = json.load(f)

    print(f"Processing {len(records)} attractor records...")
    for i, rec in enumerate(records, 1):
        attractor_synset: str = rec["attractor_synset"]
        attractor_label: str = rec["attractor_label"]
        setting: dict = rec["setting"]
        group: str = setting["group"]
        corruption: str = setting["corruption"]
        severity: int = setting["severity"]
        models: list[str] = rec["models"]
        sources: list[dict] = rec["sources"]

        slug = slugify(attractor_label)
        out_dir = OUT_DIR / corruption / str(severity) / slug

        # Attractor table
        attr_table = build_attractor_table(attractor_synset, models, group, corruption)
        write_pair(
            out_dir,
            "attractor",
            attr_table,
            caption=f"{attractor_label} ({attractor_synset}) — attractor accuracy | {corruption} s{severity}",
            label=f"{slug}__{corruption}__s{severity}__attractor",
        )

        # Sources table
        src_table = build_sources_table(sources, models, group, corruption)
        write_pair(
            out_dir,
            "sources",
            src_table,
            caption=f"Sources flowing into {attractor_label} | {corruption} s{severity}",
            label=f"{slug}__{corruption}__s{severity}__sources",
        )

        if i % 50 == 0 or i == len(records):
            print(f"  {i}/{len(records)}")

    print(f"\nDone. Tables written to:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
