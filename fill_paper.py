"""Fill the LaTeX paper from real experiment results (no numbers are invented).

  python fill_paper.py --field_images 480 --field_classes 12 --gpu "Tesla T4 (Colab)" --device "CPU, Intel i5"

Reads runs/A?_*_s<seed>/metrics.json (written by train.py / evaluate.py) and writes
paper/numbers.tex and paper/results_rows.tex. Run train.py with --ood_dir (field images)
for every variant first, otherwise the field columns cannot be computed."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(runs_dir="runs"):
    """-> {'A0': [metrics, ...per seed], ...}"""
    groups = defaultdict(list)
    for f in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = re.match(r"^(A\d)_.*_s(\d+)$", f.parent.name)
        if m:
            groups[m.group(1)].append(json.loads(f.read_text()))
    return dict(groups)


def _acc(m, domain):
    """Runs with a crop head are reported with crop-consistent decoding when available."""
    key = {"in": "in_domain", "field": "cross_domain"}[domain]
    return m.get(key + "_hier_decode") or m.get(key)


def collect(ms):
    def col(fn):
        v = [fn(m) for m in ms]
        return np.array([x for x in v if x is not None], dtype=float)

    def metric(domain, name):
        return lambda m: (_acc(m, domain) or {}).get(name)

    ia = col(metric("in", "accuracy")) * 100
    f1 = col(metric("in", "macro_f1")) * 100
    fa = col(metric("field", "accuracy")) * 100
    return {"in": ia, "f1": f1, "field": fa,
            "gap": ia - fa if len(ia) == len(fa) and len(fa) else np.array([]),
            "params": col(lambda m: m["params"] / 1e6 if m.get("params") else None),
            "size": col(lambda m: m.get("model_size_mb")),
            "lat": col(lambda m: m.get("latency_ms_per_image"))}


def ms(a, nd=1):
    if len(a) == 0:
        raise SystemExit("Missing values: run train.py with --ood_dir for every variant and every seed.")
    return f"{a.mean():.{nd}f} $\\pm$ {a.std():.{nd}f}" if len(a) > 1 else f"{a.mean():.{nd}f}"


def sgn(x):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.1f}"


def findings(R, g, focus_a=None, focus_b=None):
    """Data-driven summary text. Every statement is computed from the measured results."""
    m = lambda k, f: float(R[k][f].mean())
    sd = lambda k, f: float(R[k][f].std())
    steps = [("transfer learning (A1 vs.\\ A0)", "A0", "A1"), ("CBAM attention (A2 vs.\\ A1)", "A1", "A2"),
             ("background randomization (A3 vs.\\ A2)", "A2", "A3"), ("mask-guided attention (A4 vs.\\ A3)", "A3", "A4"),
             ("hierarchical heads (A5 vs.\\ A4)", "A4", "A5")]
    d = [(n, m(b, "field") - m(a, "field")) for n, a, b in steps]
    best = max(d, key=lambda x: x[1])
    neg = [n for n, v in d if v < 0]
    out = []
    out.append(f"The largest change in field accuracy between consecutive variants came from {best[0]}, {sgn(best[1])} points.")
    df = m("A5", "field") - m("A1", "field")
    di = m("A5", "in") - m("A1", "in")
    out.append("Compared with the transfer-learning baseline (A1), the full model "
               f"{'raises' if df >= 0 else 'lowers'} field accuracy by {abs(df):.1f} points, "
               f"{'reduces' if m('A5', 'gap') <= m('A1', 'gap') else 'increases'} the generalization gap from "
               f"{m('A1', 'gap'):.1f} to {m('A5', 'gap'):.1f} points, and changes in-domain accuracy by {sgn(di)} points.")
    n_seeds = min(len(g["A1"]), len(g["A5"]))
    if n_seeds < 2:
        out.append("Only one seed was run, so no estimate of run-to-run variation is available.")
    elif abs(df) <= max(sd("A1", "field"), sd("A5", "field")):
        out.append(f"This field-accuracy difference is within the seed-to-seed standard deviation ({n_seeds} seeds) and should be interpreted with caution.")
    else:
        out.append(f"This field-accuracy difference is larger than the seed-to-seed standard deviation ({n_seeds} seeds); no formal significance test was performed.")
    if focus_a and focus_b:
        out.append(f"The Grad-CAM leaf-focus score is {focus_a}\\% for A1 and {focus_b}\\% for A5.")
    if neg:
        out.append("The following component(s) did not improve mean field accuracy in this setting: " + "; ".join(neg) + ".")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--field_images", required=True, help="number of field images")
    ap.add_argument("--field_classes", required=True, help="number of shared classes")
    ap.add_argument("--gpu", required=True, help="training hardware, e.g. 'NVIDIA T4 (Google Colab)'")
    ap.add_argument("--device", required=True, help="latency device, e.g. 'Intel Core i5 CPU'")
    a = ap.parse_args()

    g = load_runs(a.runs)
    need = [f"A{i}" for i in range(6)]
    miss = [k for k in need if k not in g]
    if miss:
        raise SystemExit(f"No metrics found for {miss}. Run run_ablation.sh first.")
    R = {k: collect(g[k]) for k in need}
    mean = lambda k, f: float(R[k][f].mean())

    rows = []
    for k in need:
        r = R[k]
        rows.append(" & ".join([k, ms(r["in"]), ms(r["f1"]), ms(r["field"]), ms(r["gap"]),
                                ms(r["params"], 2), ms(r["size"]), ms(r["lat"])]) + " \\\\")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_rows.tex").write_text("\\renewcommand{\\ResultsRows}{%\n" + "\n".join(rows) + "}\n")

    N = {
        "AccIn": f"{mean('A5', 'in'):.1f}", "AccField": f"{mean('A5', 'field'):.1f}",
        "AccFieldBase": f"{mean('A1', 'field'):.1f}", "SizeMB": f"{mean('A5', 'size'):.1f}",
        "ParamsM": f"{mean('A5', 'params'):.2f}", "LatencyMs": f"{mean('A5', 'lat'):.1f}",
        "FieldN": a.field_images, "FieldM": a.field_classes, "GPUName": a.gpu, "DeviceName": a.device,
        "dInA": sgn(mean("A1", "in") - mean("A0", "in")),
        "dFieldA": sgn(mean("A1", "field") - mean("A0", "field")),
        "dFieldB": sgn(mean("A2", "field") - mean("A1", "field")),
        "gapB": f"{mean('A2', 'gap'):.1f}", "gapC": f"{mean('A3', 'gap'):.1f}",
        "dFieldC": sgn(mean("A4", "field") - mean("A3", "field")),
        "dFieldD": sgn(mean("A5", "field") - mean("A4", "field")),
    }
    lf = out / "figs" / "leaf_focus.json"
    if lf.exists():
        d = json.loads(lf.read_text())
        N["FocusA"], N["FocusB"] = f"{d['A1']:.1f}", f"{d['A5']:.1f}"
    N["AutoDiscussion"] = findings(R, g, N.get("FocusA"), N.get("FocusB"))
    (out / "numbers.tex").write_text("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
    print("Wrote", out / "numbers.tex", "and", out / "results_rows.tex")
    print("Seeds per variant:", {k: len(g[k]) for k in need})


if __name__ == "__main__":
    main()
