"""Single-image inference: diagnosis + Grad-CAM + severity + knowledge-base lookup."""
import json
import re
from pathlib import Path

import keras
import numpy as np
from PIL import Image

import data as D
import models as M  # noqa: F401  (registers custom layers)
from gradcam import GradCAM, overlay
from severity import estimate_severity

CONF_THRESHOLD = 0.60


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pretty(label):
    crop, _, dis = label.partition("___")
    return crop.replace("_", " ").strip(), (dis or "").replace("_", " ").strip() or "healthy"


class Predictor:
    def __init__(self, run_dir, split_file="runs/splits.json", kb_path="disease_info.json"):
        self.model = keras.models.load_model(Path(run_dir) / "best.keras")
        self.class_names = json.loads(Path(split_file).read_text())["class_names"]
        self.c2c, _ = D.class_to_crop(self.class_names)
        kb = json.loads(Path(kb_path).read_text())
        self.disclaimer = kb.pop("_disclaimer", "")
        self.generic = kb.pop("_generic")
        self.kb = {_norm(k): v for k, v in kb.items()}
        self.cam = GradCAM(self.model)

    def analyse(self, pil_img, hier_decode=True):
        img = np.asarray(pil_img.convert("RGB").resize((D.IMG_SIZE, D.IMG_SIZE)))
        x = img[None].astype("float32")
        out = self.model(x, training=False)
        out = out if isinstance(out, dict) else {"disease": out}
        p = np.asarray(out["disease"])[0].astype("float64")
        if hier_decode and "crop" in out:
            p = p * (self.c2c == int(np.argmax(np.asarray(out["crop"])[0])))
            p = p / max(p.sum(), 1e-9)
        order = np.argsort(-p)
        idx = int(order[0])
        label = self.class_names[idx]
        crop, disease = pretty(label)
        healthy = label.lower().endswith("healthy")

        cam, _ = self.cam(x, idx)
        cam_img, _ = overlay(img, cam)
        attn_img = overlay(img, np.asarray(out["attn"])[0, ..., 0])[0] if "attn" in out else None
        pct, level, _ = estimate_severity(img)

        info = self.kb.get(_norm(label), self.generic)
        return {
            "label": label, "crop": crop, "disease": disease, "healthy": healthy,
            "confidence": float(p[idx]), "uncertain": float(p[idx]) < CONF_THRESHOLD,
            "top3": [(pretty(self.class_names[i]), float(p[i])) for i in order[:3]],
            "severity_pct": 0.0 if healthy else pct, "severity_level": "None" if healthy else level,
            "image": Image.fromarray(img), "gradcam": cam_img, "attention": attn_img,
            "info": info, "disclaimer": self.disclaimer,
        }
