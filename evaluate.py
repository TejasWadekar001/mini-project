"""Evaluation: accuracy, macro precision/recall/F1, confusion matrix,
crop-consistent (hierarchical) decoding, cross-domain test, size and latency."""
import argparse
import json
import time
from pathlib import Path

import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import data as D
import models as M  # noqa: F401  (registers custom layers for load_model)


def predict(model, ds, keys):
    out = model.predict(ds, verbose=0)
    if not isinstance(out, dict):
        out = {keys[0]: out} if len(keys) == 1 else dict(zip(keys, out))
    y = np.concatenate([t["disease"].numpy() for _, t in ds])
    return out, y


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": float((y == p).mean()), "macro_precision": float(pr),
            "macro_recall": float(rc), "macro_f1": float(f1)}


def hier_decode(p_disease, p_crop, c2c):
    """Only allow diseases that belong to the predicted crop."""
    allowed = c2c[None, :] == p_crop.argmax(1)[:, None]
    return (p_disease * allowed).argmax(1)


def latency_ms(model, n=30):
    x = np.random.rand(1, D.IMG_SIZE, D.IMG_SIZE, 3).astype("float32") * 255
    model(x, training=False)
    t = time.perf_counter()
    for _ in range(n):
        model(x, training=False)
    return (time.perf_counter() - t) / n * 1000


def plot_confusion(cm, out):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run_eval(model, test_ds, keys, class_names, c2c, run_dir, ood_ds=None):
    run_dir = Path(run_dir)
    out, y = predict(model, test_ds, keys)
    pred = out["disease"].argmax(1)
    res = {"in_domain": scores(y, pred)}

    if "crop" in out:
        res["in_domain_hier_decode"] = scores(y, hier_decode(out["disease"], out["crop"], c2c))
        res["crop_accuracy"] = float((c2c[y] == out["crop"].argmax(1)).mean())

    labels = list(range(len(class_names)))
    (run_dir / "classification_report.txt").write_text(
        classification_report(y, pred, labels=labels, target_names=class_names, zero_division=0))
    plot_confusion(confusion_matrix(y, pred, labels=labels), run_dir / "confusion_matrix.png")

    if ood_ds is not None:
        o, yo = predict(model, ood_ds, keys)
        res["cross_domain"] = scores(yo, o["disease"].argmax(1))
        if "crop" in o:
            res["cross_domain_hier_decode"] = scores(yo, hier_decode(o["disease"], o["crop"], c2c))
        res["generalisation_gap_acc"] = res["in_domain"]["accuracy"] - res["cross_domain"]["accuracy"]

    res["params"] = int(model.count_params())
    res["latency_ms_per_image"] = latency_ms(model)
    f = run_dir / "best.keras"
    if f.exists():
        res["model_size_mb"] = round(f.stat().st_size / 1e6, 2)
    (run_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_file", default="runs/splits.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ood_dir", default=None)
    ap.add_argument("--ood_map", default=None, help="JSON {folder_name: class_name}")
    a = ap.parse_args()

    run = Path("runs") / a.run_name
    model = keras.models.load_model(run / "best.keras")
    keys = list(model.output_names) if not isinstance(model.output, dict) else list(model.output.keys())
    sp = json.loads(Path(a.split_file).read_text())
    names = sp["class_names"]
    c2c, _ = D.class_to_crop(names)
    f, l, c = D.to_arrays(a.data_dir, sp["test"], c2c)
    test = D.build_dataset(f, l, c, keys, a.batch)
    ood = None
    if a.ood_dir:
        mp = json.loads(Path(a.ood_map).read_text()) if a.ood_map else None
        items = D.scan_ood(a.ood_dir, names, mp)
        f, l, c = D.to_arrays(a.ood_dir, items, c2c, absolute=True)
        ood = D.build_dataset(f, l, c, keys, a.batch)
    run_eval(model, test, keys, names, c2c, run, ood)


if __name__ == "__main__":
    main()
