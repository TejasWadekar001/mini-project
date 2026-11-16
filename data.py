"""Data pipeline.

* stratified 70/15/15 splits (saved to JSON so every experiment uses the same split)
* leaf-mask estimation from colour saturation (no manual annotation needed)
* geometric / photometric augmentation that keeps image and mask aligned
* Saturation-based Background Randomization (SBR): the novel augmentation that
  replaces the (clean, lab) background with random colour/noise so the model
  cannot rely on background cues -> better generalisation to field images.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
FEAT = IMG_SIZE // 32          # spatial size of the backbone feature map (7)
EXTS = {".jpg", ".jpeg", ".png"}
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# file handling / splits
# ----------------------------------------------------------------------------
def crop_of(class_name):
    """'Tomato___Early_blight' -> 'Tomato' (PlantVillage naming)."""
    return class_name.split("___")[0]


def class_to_crop(class_names):
    crops = sorted({crop_of(c) for c in class_names})
    cmap = {c: i for i, c in enumerate(crops)}
    return np.array([cmap[crop_of(c)] for c in class_names], dtype=np.int32), crops


def scan_dataset(data_dir, max_per_class=None):
    root = Path(data_dir)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    items = []
    for i, c in enumerate(class_names):
        files = sorted(p for p in (root / c).rglob("*") if p.suffix.lower() in EXTS)
        if max_per_class:
            files = files[:max_per_class]
        items += [(str(p.relative_to(root)), i) for p in files]
    return class_names, items


def get_splits(data_dir, split_file, seed=42, max_per_class=None):
    split_file = Path(split_file)
    if split_file.exists():
        return json.loads(split_file.read_text())
    class_names, items = scan_dataset(data_dir, max_per_class)
    paths = [p for p, _ in items]
    labels = [int(l) for _, l in items]
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    out = {
        "class_names": class_names,
        "train": list(zip(p_tr, y_tr)),
        "val": list(zip(p_va, y_va)),
        "test": list(zip(p_te, y_te)),
    }
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(out))
    return out


def to_arrays(data_dir, items, c2c, absolute=False):
    files = [p if absolute else str(Path(data_dir) / p) for p, _ in items]
    labels = np.array([l for _, l in items], dtype=np.int32)
    return files, labels, c2c[labels]


def scan_ood(ood_dir, class_names, mapping=None):
    """Cross-domain (field) images. Folder names must match the training class
    names, or be translated through `mapping` {folder_name: class_name}."""
    idx = {c: i for i, c in enumerate(class_names)}
    mapping = mapping or {}
    items = []
    for d in sorted(Path(ood_dir).iterdir()):
        name = mapping.get(d.name, d.name)
        if d.is_dir() and name in idx:
            items += [(str(p), idx[name]) for p in d.rglob("*")
                      if p.suffix.lower() in EXTS]
    return items


# ----------------------------------------------------------------------------
# image ops (all batched: x is [B, H, W, 3] in 0..255)
# ----------------------------------------------------------------------------
def leaf_mask(x):
    """Estimate the leaf region: lab backgrounds are grey/black/white
    (low saturation) while leaves, including brown lesions, are saturated."""
    hsv = tf.image.rgb_to_hsv(x / 255.0)
    m = tf.cast(hsv[..., 1:2] > 0.15, tf.float32)
    m = tf.nn.max_pool2d(m, 5, 1, "SAME")        # dilate
    m = -tf.nn.max_pool2d(-m, 5, 1, "SAME")      # erode  (= morphological closing)
    return m


def randomize_background(x, mask, p=0.5):
    """SBR: with probability p per image, replace the background by a random
    colour + Gaussian noise. Edges are softened to avoid a cut-out artefact."""
    b = tf.shape(x)[0]
    color = tf.random.uniform([b, 1, 1, 3], 0.0, 255.0)
    noise = tf.random.normal(tf.shape(x), 0.0, 25.0)
    bg = tf.clip_by_value(color + noise, 0.0, 255.0)
    soft = tf.nn.avg_pool2d(mask, 5, 1, "SAME")
    swapped = soft * x + (1.0 - soft) * bg
    use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < p, tf.float32)
    return use * swapped + (1.0 - use) * x


def _make_prepare(keys, train, use_sbr):
    flip = layers.RandomFlip("horizontal_and_vertical")
    rot = layers.RandomRotation(0.1, fill_mode="reflect")
    zoom = layers.RandomZoom(0.15, fill_mode="reflect")

    def fn(x, y, c):
        mask = leaf_mask(x)
        if train:
            xm = tf.concat([x, mask * 255.0], axis=-1)       # keep mask aligned
            xm = zoom(rot(flip(xm, training=True), training=True), training=True)
            x = tf.clip_by_value(xm[..., :3], 0.0, 255.0)
            mask = tf.clip_by_value(xm[..., 3:] / 255.0, 0.0, 1.0)
            if use_sbr:
                x = randomize_background(x, mask)
            b = tf.shape(x)[0]
            gain = tf.random.uniform([b, 1, 1, 1], 0.8, 1.2)
            bias = tf.random.uniform([b, 1, 1, 1], -20.0, 20.0)
            x = tf.clip_by_value(x * gain + bias, 0.0, 255.0)
        targets = {"disease": y}
        if "crop" in keys:
            targets["crop"] = c
        if "attn" in keys:
            targets["attn"] = tf.image.resize(mask, (FEAT, FEAT), method="area")
        return x, targets

    return fn


def _load(path, y, c):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32), y, c


def build_dataset(files, labels, crop_ids, keys, batch=32, train=False,
                  use_sbr=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((files, labels, crop_ids))
    if train:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE).batch(batch)
    ds = ds.map(_make_prepare(keys, train, use_sbr), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
