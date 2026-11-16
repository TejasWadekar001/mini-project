"""Grad-CAM: which image regions pushed the model towards its prediction."""
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from matplotlib import cm
from PIL import Image


def _disease_output(model):
    out = model.output
    return out["disease"] if isinstance(out, dict) else out


def _feature_layer(model):
    for l in reversed(model.layers):          # nested backbone (transfer models)
        if isinstance(l, keras.Model):
            return l
    for l in reversed(model.layers):          # last conv layer (baseline CNN)
        if isinstance(l, layers.Conv2D):
            return l
    raise ValueError("No convolutional feature layer found")


class GradCAM:
    def __init__(self, model):
        self.gm = keras.Model(model.inputs, [_feature_layer(model).output, _disease_output(model)])

    def __call__(self, x, class_idx=None):
        """x: float array [1, H, W, 3] in 0..255. Returns (cam in 0..1, class_idx)."""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            fmap, preds = self.gm(x, training=False)
            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]
        grads = tape.gradient(score, fmap)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))[0].numpy()
        return cam / (cam.max() + 1e-8), class_idx


def overlay(img_uint8, cam, alpha=0.45):
    """Blend a small heat map (any size, 0..1) over the image. Returns (PIL image, upsampled map)."""
    h, w = img_uint8.shape[:2]
    big = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((w, h), Image.BICUBIC)) / 255.0
    heat = (cm.jet(big)[..., :3] * 255).astype("float32")
    blended = (1 - alpha) * img_uint8.astype("float32") + alpha * heat
    return Image.fromarray(blended.astype("uint8")), big
