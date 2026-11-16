"""Affected-leaf-area estimate (no TensorFlow needed).

Leaf = saturated pixels (grey/white/black backgrounds are removed).
Lesion = leaf pixels that are not 'green' (brown, yellow, grey tissue).
This is an indicative colour-based estimate, not a clinical measurement."""
import numpy as np
from PIL import Image, ImageFilter


def leaf_mask(img_uint8):
    sat = np.asarray(Image.fromarray(img_uint8).convert("HSV"))[..., 1]
    m = Image.fromarray(((sat > 38) * 255).astype("uint8"))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    return np.asarray(m) > 127


def estimate_severity(img_uint8):
    """Returns (percent_affected, level, lesion_mask)."""
    leaf = leaf_mask(img_uint8)
    r, g, b = (img_uint8[..., i].astype(float) for i in range(3))
    green = (g > r * 1.05) & (g > b)
    lesion = leaf & ~green
    pct = 100.0 * lesion.sum() / max(int(leaf.sum()), 1)
    level = "Mild" if pct < 10 else "Moderate" if pct < 30 else "Severe"
    return pct, level, lesion
