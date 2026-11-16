"""Collect runs/*/metrics.json into the ablation table for the paper (mean +- std over seeds).
   python make_results_table.py        -> results_table.md and results_table.csv"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

groups = defaultdict(list)
for f in sorted(Path("runs").glob("*/metrics.json")):
    name = re.sub(r"_s\d+$", "", f.parent.name)
    groups[name].append(json.loads(f.read_text()))


def ms(vals, pct=True, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    k = 100 if pct else 1
    return f"{np.mean(vals) * k:.{nd}f} ± {np.std(vals) * k:.{nd}f}" if len(vals) > 1 else f"{vals[0] * k:.{nd}f}"


def pick(m, *path):
    for p in path:
        m = m.get(p) if isinstance(m, dict) else None
    return m


head = ["Run", "Seeds", "In-domain acc (%)", "In-domain macro-F1 (%)", "Field acc (%)", "Gap (pts)", "Params (M)", "Size (MB)", "Latency (ms)"]
rows = []
for name, ms_list in groups.items():
    rows.append([
        name, str(len(ms_list)),
        ms([pick(m, "in_domain", "accuracy") for m in ms_list]),
        ms([pick(m, "in_domain", "macro_f1") for m in ms_list]),
        ms([pick(m, "cross_domain", "accuracy") for m in ms_list]),
        ms([None if m.get("generalisation_gap_acc") is None else m["generalisation_gap_acc"] for m in ms_list]),
        ms([None if m.get("params") is None else m["params"] / 1e6 for m in ms_list], pct=False),
        ms([m.get("model_size_mb") for m in ms_list], pct=False, nd=1),
        ms([m.get("latency_ms_per_image") for m in ms_list], pct=False, nd=1),
    ])
md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
Path("results_table.md").write_text(md + "\n")
with open("results_table.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(head)
    w.writerows(rows)
print(md)
