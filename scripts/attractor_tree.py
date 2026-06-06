from __future__ import annotations
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ATTRACTORS_JSON = Path("results/representations/attractors/attractors.json")
IMAGE_DIR = Path("results/imagenet_samples")
OUT_DIR = Path("results/representations/attractors/trees_by_corruption")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPEG"]
IMG_W, IMG_H = 90, 90
N_MODELS = 15
MIN_SOURCES = 1
TOP_N_PER_CORRUPTION = None
RENDER_FORMAT = "png"


def find_image(synset):
    for ext in IMG_EXTS:
        p = IMAGE_DIR / f"{synset}{ext}"
        if p.exists():
            return p.as_posix()
    return None


def aggregate_by_corruption(records):
    out = {}
    for record in records:
        corruption = record["setting"]["corruption"]
        severity = record["setting"]["severity"]
        a = record["attractor_synset"]
        atts = out.setdefault(corruption, {})
        node = atts.setdefault(a, {"label": record["attractor_label"], "severities": set(), "sources": {}})
        node["severities"].add(severity)
        for source in record.get("sources", []):
            sn = node["sources"].setdefault(source["synset"],
                                            {"label": source["label"], "severities": set(),
                                             "deltas": [], "models": set()})
            sn["severities"].add(severity)
            sn["models"].update(source.get("models", []))
            if source.get("mean_delta") is not None:
                sn["deltas"].append(float(source["mean_delta"]))
    return out


def html_label(label, synset):
    name = html.escape(label.replace("_", " "))
    cap = f"{name} ({synset})"
    img = find_image(synset)
    rows = ""
    if img:
        rows += (f'<TR><TD FIXEDSIZE="TRUE" WIDTH="{IMG_W}" HEIGHT="{IMG_H}">'
                 f'<IMG SRC="{img}" SCALE="TRUE"/></TD></TR>')
    rows += f'<TR><TD>{cap}</TD></TR>'
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">{rows}</TABLE>>'


def edge_width(sdata):
    n_models = len(sdata["models"])
    if n_models:
        return 1.0 + 3.0 * (n_models / N_MODELS)
    md = sum(sdata["deltas"]) / len(sdata["deltas"]) if sdata["deltas"] else 0.0
    return 1.0 + 3.0 * md


def build_dot(corruption, synset, node):
    title = f"{corruption} (severity aggregated)"
    lines = [f'digraph "{corruption}_{synset}" {{',
             "  rankdir=LR;",
             "  splines=ortho;",
             "  nodesep=0.22; ranksep=0.55;",
             f'  labelloc="t"; label="{title}"; fontname="Helvetica";',
             "  node [shape=plaintext, fontname=\"Helvetica\"];",
             "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#888888\"];",
             f'  "{synset}" [label={html_label(node["label"], synset)}];']
    for sn, sdata in sorted(node["sources"].items(), key=lambda kv: len(kv[1]["severities"]), reverse=True):
        sev = ",".join(str(s) for s in sorted(sdata["severities"]))
        lines.append(f'  "{sn}" [label={html_label(sdata["label"], sn)}];')
        lines.append(f'  "{synset}" -> "{sn}" [xlabel="s{sev}", penwidth={edge_width(sdata):.2f}];')
    lines.append("}")
    return "\n".join(lines)


def safe(name):
    return re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")


def main():
    records = json.loads(ATTRACTORS_JSON.read_text())
    by_corr = aggregate_by_corruption(records)
    has_dot = shutil.which("dot") is not None
    made = 0
    for corruption, attractors in sorted(by_corr.items()):
        items = sorted(attractors.items(), key=lambda kv: len(kv[1]["sources"]), reverse=True)
        items = [it for it in items if len(it[1]["sources"]) >= MIN_SOURCES]
        if TOP_N_PER_CORRUPTION is not None:
            items = items[:TOP_N_PER_CORRUPTION]
        for synset, node in items:
            stem = f"{safe(corruption)}_{synset}_{safe(node['label'])}"
            dot_path = OUT_DIR / f"{stem}.dot"
            dot_path.write_text(build_dot(corruption, synset, node), encoding="utf-8")
            if has_dot:
                out = OUT_DIR / f"{stem}.{RENDER_FORMAT}"
                subprocess.run(["dot", f"-T{RENDER_FORMAT}", dot_path.as_posix(), "-o", out.as_posix()], check=True)
            made += 1
    print(f"{made} corruption-attractor trees -> {OUT_DIR}")
    if not has_dot:
        print(f"dot not found; render each with: dot -T{RENDER_FORMAT} <file>.dot -o <file>.{RENDER_FORMAT}")


if __name__ == "__main__":
    main()