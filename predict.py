"""Batch / field testing from the command line.
   python predict.py --run_name A5_full --input field_photos/ --out_dir reports/
Writes one PDF report per image and predictions.csv."""
import argparse
import csv
from pathlib import Path

from PIL import Image

from infer import Predictor
from report import build_report

ap = argparse.ArgumentParser()
ap.add_argument("--run_name", required=True)
ap.add_argument("--input", required=True, help="image file or folder")
ap.add_argument("--out_dir", default="reports")
ap.add_argument("--split_file", default="runs/splits.json")
ap.add_argument("--no_hier", action="store_true")
a = ap.parse_args()

p = Path(a.input)
files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
pred = Predictor(Path("runs") / a.run_name, a.split_file)

with open(out / "predictions.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "crop", "condition", "confidence", "uncertain", "severity_level", "severity_pct"])
    for f in files:
        r = pred.analyse(Image.open(f), hier_decode=not a.no_hier)
        build_report(str(out / f"{f.stem}_report.pdf"), r, plant_note=f.name)
        w.writerow([f.name, r["crop"], r["disease"], f"{r['confidence']:.3f}", r["uncertain"],
                    r["severity_level"], f"{r['severity_pct']:.1f}"])
        print(f"{f.name}: {r['crop']} / {r['disease']} ({r['confidence'] * 100:.1f}%)")
print("Done ->", out)
