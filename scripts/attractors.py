from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from constants import IMAGENET_C_CORRUPTION_GROUPS
from fragile.definitions import DEFINITIONS
from fragile.experiments import get_dfs_for_all_models 
from fragile.fragile import get_absolute_fragile, get_relative_drop_fragile
from space import CorruptionVariations  

PRED_DIR = ROOT / "results"
OUT_DIR = ROOT / "results" / "representations" / "attractors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "resnet50", "resnet152", "regnet_y_16gf", "resnext101_64x4d",
    "wide_resnet50_2", "wide_resnet101_2", "efficientnet_b4",
    "efficientnet_v2_m", "vit_b_16", "vit_l_16", "swin_b",
    "swin_v2_b", "maxvit_t", "convnext_base", "convnext_large",
]

GROUPS = [g for g in IMAGENET_C_CORRUPTION_GROUPS if g != "extra"]
SEVERITIES = [1, 2, 3]

TAU = 0.3   
ROBUST_CORRUPT_ATTR = 0.6  
COMMON_ATTR_MIN_MODELS = 3

AB = DEFINITIONS["ab"]

GROUP_OF = {
    corruption: group
    for group, corruptions in IMAGENET_C_CORRUPTION_GROUPS.items()
    for corruption in corruptions
}


def load_label_maps():
    with open(ROOT / "imagenet_class_index.json") as f:
        index = json.load(f)
    idx_to_synset = {int(i): syn for i, (syn, _) in index.items()}
    synset_to_label = {syn: label for syn, label in index.values()}
    return idx_to_synset, synset_to_label


def clean_pred_path(model: str) -> Path:
    return PRED_DIR / f"{model}_imagenet.csv"


def cond_pred_path(model: str, group: str, corruption: str, severity: int) -> Path:
    return PRED_DIR / f"{model}_imagenet_c_{group}_{corruption}_{severity}.csv"


def flow(path: Path, idx_to_synset: dict[int, str]) -> pd.DataFrame:
    """Per source synset, the fraction of its images predicted as each target synset.

    y_pred is an ImageNet index (0-999); map it to a synset so source and target live in
    the same namespace. Returns columns [source, target, frac].
    """
    p = pd.read_csv(path, usecols=["synset", "y_pred"])
    p["target"] = p["y_pred"].map(idx_to_synset)
    p = p.dropna(subset=["target"])
    n = p.groupby(["synset", "target"]).size().rename("n").reset_index()
    tot = p.groupby("synset").size().rename("tot")
    n = n.merge(tot, on="synset")
    n["frac"] = n["n"] / n["tot"]
    return n.rename(columns={"synset": "source"})[["source", "target", "frac"]]


def model_edges(
    model: str,
    group: str,
    corruption: str,
    severity: int,
    clean_flow: pd.DataFrame,
    fragile_synsets: set[str],
    acc_corrupt: "pd.Series[float]",
    idx_to_synset: dict[int, str],
) -> pd.DataFrame:
    """Valid source->attractor edges for one model under one setting.

    An edge qualifies when: source is fragile (A and B), the corruption drives >= TAU of the
    source's images into the target, and the target is robust (high corrupt accuracy and not
    itself fragile).
    """
    corr_path = cond_pred_path(model, group, corruption, severity)
    if not corr_path.exists():
        return pd.DataFrame(columns=["source", "attractor", "delta", "attractor_acc_corrupt"])

    corr = flow(corr_path, idx_to_synset).rename(columns={"frac": "corr"})
    d = clean_flow.rename(columns={"frac": "clean"}).merge(
        corr, on=["source", "target"], how="outer"
    ).fillna(0.0)
    d = d[d["source"] != d["target"]].copy()
    d["delta"] = d["corr"] - d["clean"]

    d = d[d["source"].isin(fragile_synsets) & (d["delta"] >= TAU)]
    d["attractor_acc_corrupt"] = d["target"].map(acc_corrupt).fillna(0.0)
    d = d[
        (d["attractor_acc_corrupt"] >= ROBUST_CORRUPT_ATTR)
        & ~d["target"].isin(fragile_synsets)
    ]
    return d.rename(columns={"target": "attractor"})[
        ["source", "attractor", "delta", "attractor_acc_corrupt"]
    ]


def fragile_lookup(df: pd.DataFrame) -> tuple[set[str], "pd.Series[float]"]:
    """A and B fragile synsets and per-class corrupt accuracy for one model df."""
    df = get_absolute_fragile(df)
    df = get_relative_drop_fragile(df)
    is_fragile = AB.combine(df)
    fragile_synsets = set(df.loc[is_fragile, "synset"])
    acc_corrupt = df.set_index("synset")["acc_corrupt"]
    return fragile_synsets, acc_corrupt


def analyze_setting(
    group: str,
    corruption: str,
    severity: int,
    clean_flows: dict[str, pd.DataFrame],
    idx_to_synset: dict[int, str],
    synset_to_label: dict[str, str],
) -> list[dict]:
    """Return attractor records for one (group, corruption, severity) setting."""
    var = CorruptionVariations(
        groups=[group], corruptions=[corruption], severities=[severity]
    )
    dfs = get_dfs_for_all_models(var)

    edges = []
    for model in MODELS:
        df = dfs.get(model)
        if df is None or df.empty:
            continue
        fragile_synsets, acc_corrupt = fragile_lookup(df)
        if not fragile_synsets:
            continue
        e = model_edges(
            model, group, corruption, severity,
            clean_flows[model], fragile_synsets, acc_corrupt, idx_to_synset,
        )
        if e.empty:
            continue
        e["model"] = model
        edges.append(e)

    if not edges:
        return []

    E = pd.concat(edges, ignore_index=True)

    records = []
    for attractor, grp in E.groupby("attractor"):
        models = sorted(grp["model"].unique())
        if len(models) < COMMON_ATTR_MIN_MODELS:
            continue

        sources = []
        for source, sgrp in grp.groupby("source"):
            sources.append({
                "synset": source,
                "label": synset_to_label.get(source, source),
                "mean_delta": round(float(sgrp["delta"].mean()), 4),
                "models": sorted(sgrp["model"].unique()),
            })
        sources.sort(key=lambda s: len(s["models"]) * 100 + s["mean_delta"], reverse=True)

        records.append({
            "attractor_synset": attractor,
            "attractor_label": synset_to_label.get(attractor, attractor),
            "setting": {"group": group, "corruption": corruption, "severity": severity},
            "n_models": len(models),
            "models": models,
            "attractor_mean_acc_corrupt": round(float(grp["attractor_acc_corrupt"].mean()), 4),
            "mean_inflow": round(float(grp["delta"].sum() / len(models)), 4),
            "sources": sources,
        })

    return records


def main():
    idx_to_synset, synset_to_label = load_label_maps()

    clean_flows = {}
    for model in MODELS:
        path = clean_pred_path(model)
        if path.exists():
            clean_flows[model] = flow(path, idx_to_synset)

    all_records = []
    attractor_settings: dict[str, dict] = {}
    for severity in SEVERITIES:
        for group in GROUPS:
            for corruption in IMAGENET_C_CORRUPTION_GROUPS[group]:
                recs = analyze_setting(
                    group, corruption, severity, clean_flows,
                    idx_to_synset, synset_to_label,
                )
                if recs:
                    print(
                        f"{group}/{corruption} s{severity}: "
                        f"{len(recs)} common attractor(s)"
                    )
                all_records.extend(recs)

                setting_tag = f"{corruption}_{severity}"
                for r in recs:
                    entry = attractor_settings.setdefault(
                        r["attractor_synset"],
                        {
                            "attractor_synset": r["attractor_synset"],
                            "attractor_label": r["attractor_label"],
                            "settings": [],
                        },
                    )
                    entry["settings"].append(setting_tag)

    all_records.sort(key=lambda r: (r["n_models"], r["mean_inflow"]), reverse=True)

    # Collapse per-attractor settings into a sorted, deduped list with a count.
    set_attractors = []
    for entry in attractor_settings.values():
        settings = sorted(set(entry["settings"]))
        corruptions = sorted(set(["_".join(corr.split("_")[:-1]) for corr in settings]))
        set_attractors.append({**entry, "settings": settings, "corruptions": corruptions, "n_settings": len(settings), "n_corruptions": len(corruptions)})
    set_attractors.sort(key=lambda e: e["n_settings"], reverse=True)

    out = OUT_DIR / "attractors.json"
    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)

    out_attr = OUT_DIR / "set_attractors.json"
    with open(out_attr, "w") as f:
        json.dump(set_attractors, f, indent=2)

    print(f"\n{len(all_records)} attractor records across the sweep")
    print(f"-> {out}")
    print(f"{len(set_attractors)} distinct attractor classes")
    print(f"-> {out_attr}")


if __name__ == "__main__":
    main()
