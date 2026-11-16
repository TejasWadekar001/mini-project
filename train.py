"""Train any ablation variant.

Examples
  python train.py --data_dir PV/color --run_name A0_cnn --model cnn
  python train.py --data_dir PV/color --run_name A1_mnv2 --model transfer
  python train.py --data_dir PV/color --run_name A5_full --model transfer \
      --cbam --sbr --attn_sup --hier
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
import evaluate as E
import models as M


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--backbone", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    ap.add_argument("--cbam", action="store_true")
    ap.add_argument("--sbr", action="store_true", help="background randomization")
    ap.add_argument("--attn_sup", action="store_true", help="mask-guided attention loss")
    ap.add_argument("--hier", action="store_true", help="crop + disease heads")
    ap.add_argument("--epochs", type=int, default=30, help="cnn only")
    ap.add_argument("--epochs_head", type=int, default=8)
    ap.add_argument("--epochs_ft", type=int, default=12)
    ap.add_argument("--unfreeze_last", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_file", default="runs/splits.json")
    ap.add_argument("--max_per_class", type=int, default=None, help="for quick tests")
    ap.add_argument("--no_pretrained", action="store_true", help="offline smoke test")
    ap.add_argument("--ood_dir", default=None)
    ap.add_argument("--ood_map", default=None)
    return ap.parse_args()


def callbacks(run, patience):
    return [
        keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2, min_lr=1e-7),
        keras.callbacks.CSVLogger(run / "log.csv", append=True),
    ]


def plot_history(h, out):
    acc = next((k for k in h if k.endswith("accuracy") and not k.startswith("val")
                and ("disease" in k or k == "accuracy")), None)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    if acc:
        ax[1].plot(h[acc], label="train")
        ax[1].plot(h["val_" + acc], label="val")
        ax[1].set_title("Disease accuracy")
        ax[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    a = parse()
    keras.utils.set_random_seed(a.seed)
    run = Path("runs") / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))

    sp = D.get_splits(a.data_dir, a.split_file, a.seed, a.max_per_class)
    names = sp["class_names"]
    c2c, crops = D.class_to_crop(names)

    use_extra = a.model == "transfer"
    keys = M.output_keys(a.hier and use_extra, a.attn_sup and use_extra)
    ds = {}
    for s in ("train", "val", "test"):
        f, l, c = D.to_arrays(a.data_dir, sp[s], c2c)
        ds[s] = D.build_dataset(f, l, c, keys, a.batch, train=(s == "train"),
                                use_sbr=a.sbr and s == "train", seed=a.seed)

    model = M.build_model(a.model, len(names), len(crops), backbone=a.backbone,
                          cbam=a.cbam, hier=a.hier, attn_sup=a.attn_sup,
                          pretrained=not a.no_pretrained) if use_extra \
        else M.build_model("cnn", len(names), len(crops))
    model.summary(print_fn=lambda s: None)

    hist = {}

    def fit(epochs, patience):
        h = model.fit(ds["train"], validation_data=ds["val"], epochs=epochs,
                      callbacks=callbacks(run, patience), verbose=2).history
        for k, v in h.items():
            hist.setdefault(k, []).extend(float(x) for x in v)

    if a.model == "cnn":
        M.compile_model(model, keys, a.lr)
        fit(a.epochs, 6)
    else:
        M.set_backbone(model, False)                       # stage 1: train the head
        M.compile_model(model, keys, a.lr)
        fit(a.epochs_head, 3)
        M.set_backbone(model, True, a.unfreeze_last)       # stage 2: fine-tune top layers
        M.compile_model(model, keys, a.lr_ft)
        fit(a.epochs_ft, 4)

    (run / "history.json").write_text(json.dumps(hist))
    plot_history(hist, run / "history.png")
    model.save(run / "best.keras")

    ood = None
    if a.ood_dir:
        mp = json.loads(Path(a.ood_map).read_text()) if a.ood_map else None
        items = D.scan_ood(a.ood_dir, names, mp)
        f, l, c = D.to_arrays(a.ood_dir, items, c2c, absolute=True)
        ood = D.build_dataset(f, l, c, keys, a.batch)
    E.run_eval(model, ds["test"], keys, names, c2c, run, ood)


if __name__ == "__main__":
    main()
