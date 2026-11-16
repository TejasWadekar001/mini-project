"""Models.

kind="cnn"       : baseline CNN trained from scratch (A0)
kind="transfer"  : ImageNet backbone + GAP + softmax head (A1)
   + cbam        : CBAM attention on the backbone feature map (A2)
   + attn_sup    : Mask-Guided Attention supervision - the CBAM spatial map is
                   trained to match the estimated leaf mask (novel, A4)
   + hier        : hierarchical multi-task heads (crop + disease) (A5)
"""
import keras
from keras import layers

from data import IMG_SIZE


@keras.saving.register_keras_serializable(package="plantdx")
class ChannelPool(layers.Layer):
    """[mean, max] over channels -> input of CBAM's spatial attention."""

    def call(self, x):
        ops = keras.ops
        return ops.concatenate([ops.mean(x, axis=-1, keepdims=True),
                                ops.max(x, axis=-1, keepdims=True)], axis=-1)

    def compute_output_shape(self, shape):
        return (*shape[:-1], 2)


def output_keys(hier=False, attn_sup=False):
    return ["disease"] + (["crop"] if hier else []) + (["attn"] if attn_sup else [])


def cbam_block(x, ratio=16, name="cbam"):
    """Channel attention followed by spatial attention. Returns (features, spatial_map)."""
    ch = x.shape[-1]
    mlp1 = layers.Dense(max(ch // ratio, 8), activation="relu", name=f"{name}_mlp1")
    mlp2 = layers.Dense(ch, name=f"{name}_mlp2")
    avg = mlp2(mlp1(layers.GlobalAveragePooling2D(keepdims=True)(x)))
    mx = mlp2(mlp1(layers.GlobalMaxPooling2D(keepdims=True)(x)))
    ca = layers.Activation("sigmoid")(layers.Add()([avg, mx]))
    x = layers.Multiply()([x, ca])
    sa = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                       name=f"{name}_spatial")(ChannelPool()(x))
    return layers.Multiply()([x, sa]), sa


def build_cnn(num_classes):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="disease")(x)
    return keras.Model(inp, {"disease": out}, name="baseline_cnn")


def build_transfer(num_classes, num_crops, backbone="mobilenetv2", cbam=False,
                   hier=False, attn_sup=False, pretrained=True):
    if attn_sup and not cbam:
        raise ValueError("--attn_sup needs --cbam")
    weights = "imagenet" if pretrained else None
    shape = (IMG_SIZE, IMG_SIZE, 3)
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                              input_shape=shape)
        pre = layers.Rescaling(1 / 127.5, offset=-1)
    elif backbone == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                                 input_shape=shape)
        pre = None                                 # EfficientNet rescales internally
    else:
        raise ValueError(backbone)

    inp = keras.Input(shape)
    x = pre(inp) if pre is not None else inp
    f = base(x, training=False)                     # BatchNorm stays in inference mode
    outs = {}
    if cbam:
        f, sa = cbam_block(f)
        if attn_sup:
            outs["attn"] = layers.Activation("linear", name="attn")(sa)
    g = layers.Dropout(0.3)(layers.GlobalAveragePooling2D()(f))
    outs["disease"] = layers.Dense(num_classes, activation="softmax", name="disease")(g)
    if hier:
        outs["crop"] = layers.Dense(num_crops, activation="softmax", name="crop")(g)
    return keras.Model(inp, outs, name=f"{backbone}{'_cbam' if cbam else ''}")


def build_model(kind, num_classes, num_crops, **kw):
    return build_cnn(num_classes) if kind == "cnn" else build_transfer(num_classes, num_crops, **kw)


def set_backbone(model, trainable, unfreeze_last=40):
    """Stage 1: freeze backbone. Stage 2: unfreeze the last N layers (BN stays frozen)."""
    base = next((l for l in model.layers if isinstance(l, keras.Model)), None)
    if base is None:
        return
    if not trainable:
        base.trainable = False
        return
    base.trainable = True
    n = len(base.layers)
    for i, l in enumerate(base.layers):
        l.trainable = i >= n - unfreeze_last and not isinstance(l, layers.BatchNormalization)


def compile_model(model, keys, lr):
    loss = {k: ("binary_crossentropy" if k == "attn" else "sparse_categorical_crossentropy")
            for k in keys}
    w = {"disease": 1.0, "crop": 0.3, "attn": 0.2}
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        loss_weights={k: w[k] for k in keys},
        metrics={k: ["accuracy"] for k in keys if k != "attn"},
    )
