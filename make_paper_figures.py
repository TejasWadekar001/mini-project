"""Create the data-dependent paper figures from YOUR real data and trained models.

  python make_paper_figures.py --data_dir PlantVillage/color --field_dir path/to/field_images

Writes paper/figs/: fig_sbr_examples.png, fig_confusion.png, fig_ablation.png,
fig_gradcam.png and leaf_focus.json (Grad-CAM leaf-focus score, used by fill_paper.py).
Needs: split file + runs from training. The Grad-CAM part needs TensorFlow."""
import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from fill_paper import collect, load_runs
from severity import leaf_mask

IMG = 224


def sbr_numpy(x, mask, rng, sigma=25.0):
    """NumPy version of data.randomize_background (same maths, p=1)."""
    soft = np.asarray(Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.BoxBlur(2))) / 255.0
    soft = soft[..., None]
    bg = np.clip(rng.uniform(0, 255, (1, 1, 3)) + rng.normal(0, sigma, x.shape), 0, 255)
    return (soft * x + (1 - soft) * bg).astype("uint8")


def fig_sbr(a, out):
    sp = json.loads(Path(a.split_file).read_text())
    random.seed(1)
    items = random.sample(sp["test"], 4)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(4, 4, figsize=(6.5, 6.5))
    for c, t in enumerate(["Original", "Leaf mask", "SBR variant 1", "SBR variant 2"]):
        ax[0, c].set_title(t, fontsize=9)
    for r, (rel, _) in enumerate(items):
        x = np.asarray(Image.open(Path(a.data_dir) / rel).convert("RGB").resize((IMG, IMG)))
        m = leaf_mask(x)
        for c, im in enumerate([x, m, sbr_numpy(x, m, rng), sbr_numpy(x, m, rng)]):
            ax[r, c].imshow(im, cmap="gray" if c == 1 else None)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_sbr_examples.png", dpi=200)
    plt.close(fig)


def fig_ablation(a, out):
    g = load_runs(a.runs)
    keys = [f"A{i}" for i in range(6)]
    R = {k: collect(g[k]) for k in keys if k in g}
    x = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(x - 0.2, [R[k]["in"].mean() for k in R], 0.4, yerr=[R[k]["in"].std() for k in R], label="In-domain", color="#2E7D32")
    ax.bar(x + 0.2, [R[k]["field"].mean() for k in R], 0.4, yerr=[R[k]["field"].std() for k in R], label="Field", color="#F9A825")
    ax.set_xticks(x)
    ax.set_xticklabels(list(R))
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation.png", dpi=250)
    plt.close(fig)


def fig_gradcam(a, out):
    import keras
    import models  # noqa: F401  registers custom layers
    from gradcam import GradCAM, overlay

    files = sorted(f for f in Path(a.field_dir).rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.seed(2)
    files = random.sample(files, min(len(files), a.max_images))
    nets = {k: keras.models.load_model(Path(a.runs) / r / "best.keras") for k, r in
            (("A1", a.baseline), ("A5", a.full))}
    cams = {k: GradCAM(m) for k, m in nets.items()}
    focus = {"A1": [], "A5": []}
    shown = []
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB").resize((IMG, IMG)))
        x = img[None].astype("float32")
        mask = leaf_mask(img).astype("float32")
        row = [img]
        for k in ("A1", "A5"):
            cam, _ = cams[k](x)
            ov, big = overlay(img, cam)
            focus[k].append(float((big * mask).sum() / (big.sum() + 1e-8)))
            row.append(np.asarray(ov))
        if len(shown) < a.n:
            shown.append(row)
    fig, ax = plt.subplots(len(shown), 3, figsize=(5.2, 1.8 * len(shown)))
    ax = np.atleast_2d(ax)
    for c, t in enumerate(["Field image", "A1 Grad-CAM", "A5 Grad-CAM"]):
        ax[0, c].set_title(t, fontsize=8)
    for r, row in enumerate(shown):
        for c, im in enumerate(row):
            ax[r, c].imshow(im)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_gradcam.png", dpi=200)
    plt.close(fig)
    (out / "leaf_focus.json").write_text(json.dumps({k: 100 * float(np.mean(v)) for k, v in focus.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--field_dir", default=None)
    ap.add_argument("--split_file", default="runs/splits.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper/figs")
    ap.add_argument("--baseline", default="A1_mnv2_s42")
    ap.add_argument("--full", default="A5_full_s42")
    ap.add_argument("--n", type=int, default=4, help="images shown in the Grad-CAM figure")
    ap.add_argument("--max_images", type=int, default=200, help="images used for the leaf-focus score")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_sbr(a, out)
    fig_ablation(a, out)
    cm = Path(a.runs) / a.full / "confusion_matrix.png"
    if cm.exists():
        shutil.copy(cm, out / "fig_confusion.png")
    if a.field_dir:
        fig_gradcam(a, out)
    print("Figures written to", out)


if __name__ == "__main__":
    main()
