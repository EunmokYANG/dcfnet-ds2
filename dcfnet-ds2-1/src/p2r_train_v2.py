"""p2r_train_v2.py -- training runs for the paper-2 resubmission.

Uses the partitions written by src.p2r_data_v2 (explicit train / val / test).
Unlike src.train, the validation set is read from the file, so it is
chronological on NF-UNSW-NB15-v3 and drawn after deduplication on CICIoT2023.
Optimizer, loss, schedule and early stopping are those of src.train.

Variants (name -> reviewer comment)
  m4      proposed model, maximum degree K                       (main)
  m4g     m4 + input-dependent degree gate                       (ablation)
  m4m     m4 + MLP branch D-256-128 (src.model)                  (ablation)
  nn_mlp  plain MLP D-256-256-128-C (src.model)                  (baseline)
  m2w     degree <= 2 with H parallel degree-2 heads; H = K-1 gives
          exactly the parameter count of the degree-K model      R1-4, R1-12
  mlp_s   plain MLP D-h-h-C, h chosen to match m4 at K = 4       R1-12, R2-8
  m4ms    m4 + narrow MLP branch (width W) whose logit output
          carries an L2 activity penalty LAMBDA                  R2-8

Every run writes one JSON to outputs/p2r/runs/ (resumable: an existing JSON
is skipped) holding test metrics, per-class precision / recall / PR-AUC,
parameter count, epochs, training time and inference time.

Run:
  python -m src.p2r_train_v2 --dataset ciciot2023 --tag p2r --plan stage1
  python -m src.p2r_train_v2 --dataset nfunsw --tag p2r --plan stage1 --seeds 10
  python -m src.p2r_train_v2 --dataset nfunsw --tag p2r_enc --only m4:K4 --seeds 10
"""

import argparse
import json
import os
import time

import numpy as np

from src import config as C

OUT = C.OUT_DIR / "p2r"
RUN_DIR = OUT / "runs"
MODEL_DIR = OUT / "models"
HIST_DIR = OUT / "history"
for _d in (RUN_DIR, MODEL_DIR, HIST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# stage-1 configurations: (variant, K, extra)
STAGE1 = [
    ("m4", 4, {}),
    ("nn_mlp", 4, {}),
    ("m4m", 4, {}),
    ("m4g", 4, {}),
    ("m4", 2, {}), ("m4", 3, {}), ("m4", 5, {}), ("m4", 6, {}),
    ("m2w", 2, {"heads": 3}),          # parameter-matched to m4 K = 4
    ("m2w", 2, {"heads": 5}),          # parameter-matched to m4 K = 6
    ("mlp_s", 4, {}),                  # parameter-matched to m4 K = 4
    ("m4ms", 4, {"width": 16, "lam": 1e-3}),
    ("m4ms", 4, {"width": 16, "lam": 1e-2}),
]


def run_name(stem, variant, K, extra, seed):
    s = f"{stem}_{variant}_K{K}"
    if "heads" in extra:
        s += f"_H{extra['heads']}"
    if "width" in extra:
        s += f"_W{extra['width']}_L{extra['lam']:g}"
    return f"{s}_s{seed}"


def set_seed(seed):
    """Same seeding as src.train.set_seed of the v1 pipeline."""
    import random
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_threads(n):
    if n:
        os.environ["OMP_NUM_THREADS"] = str(n)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(2)


def compile_model(model):
    import tensorflow as tf
    from src.model import focal_loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=C.LEARNING_RATE),
        loss=focal_loss(), metrics=["accuracy"])
    return model


def build_m2w(D, n_cls, heads, l2=1e-5):
    """Degree <= 2 with `heads` independent degree-2 terms.

    logit = V1 u1~ + sum_h V2h u2h~ + b,  u2h = x * (W2h u1~).
    Every summand is homogeneous of degree 1 or 2, so the degree identity
    still holds; with heads = K - 1 the parameter count equals that of the
    degree-K model, (K-1)D^2 + KCD + C + 2K.
    """
    from tensorflow.keras import layers, models, regularizers
    from src.model import BiasAdd, DegreeScale, cross_layer_pure
    x = layers.Input(shape=(D,), name="inputs")
    u1 = DegreeScale(name="scale_deg1")(x)
    parts = [layers.Dense(n_cls, use_bias=False,
                          kernel_regularizer=regularizers.l2(l2),
                          name="logit_deg1")(u1)]
    for h in range(1, heads + 1):
        u = cross_layer_pure(x, u1, l2=l2, name=f"cross_deg2_h{h}")
        un = DegreeScale(name=f"scale_deg2_h{h}")(u)
        parts.append(layers.Dense(n_cls, use_bias=False,
                                  kernel_regularizer=regularizers.l2(l2),
                                  name=f"logit_deg2_h{h}")(un))
    logits = BiasAdd(name="logits")(layers.Add(name="logit_sum")(parts))
    out = layers.Activation("softmax", name="probs")(logits)
    return compile_model(models.Model(x, out, name="dcfnet_m2w"))


def build_mlp_s(D, n_cls, target):
    """Plain MLP D-h-h-C (ReLU, BN and dropout 0.3 after each hidden layer),
    with h chosen so that the parameter count is closest to `target`."""
    from tensorflow.keras import layers, models

    def make(h):
        x = layers.Input(shape=(D,), name="inputs")
        z = x
        for _ in range(2):
            z = layers.Dense(h, activation="relu")(z)
            z = layers.BatchNormalization()(z)
            z = layers.Dropout(0.3)(z)
        out = layers.Dense(n_cls, activation="softmax", name="probs")(z)
        return models.Model(x, out, name="mlp_small")

    best = min(range(8, 257), key=lambda h: abs(
        h * h + h * (D + n_cls + 10) + n_cls - target))
    return compile_model(make(best)), best


def build_m4ms(D, n_cls, K, width, lam, l2=1e-5):
    """m4 + a narrow non-decomposable branch D-W-C whose logit output is
    penalized by lam * mean(phi_MLP^2) (Keras L2 activity regularizer)."""
    from tensorflow.keras import layers, models, regularizers
    from src.model import BiasAdd, DegreeScale, cross_layer_pure
    x = layers.Input(shape=(D,), name="inputs")
    parts, u_norm = [], None
    for k in range(1, K + 1):
        u = x if k == 1 else cross_layer_pure(x, u_norm, l2=l2,
                                              name=f"cross_deg{k}")
        u_norm = DegreeScale(name=f"scale_deg{k}")(u)
        parts.append(layers.Dense(n_cls, use_bias=False,
                                  kernel_regularizer=regularizers.l2(l2),
                                  name=f"logit_deg{k}")(u_norm))
    m = layers.Dense(width, activation="relu", name="mlp_hidden")(x)
    parts.append(layers.Dense(
        n_cls, use_bias=False, name="logit_mlp",
        activity_regularizer=regularizers.l2(lam))(m))
    logits = BiasAdd(name="logits")(layers.Add(name="logit_sum")(parts))
    out = layers.Activation("softmax", name="probs")(logits)
    return compile_model(models.Model(x, out, name="dcfnet_m4ms"))


def flops_per_flow(model):
    """Multiply-add count of one forward pass (2 per kernel weight, plus the
    elementwise products, DegreeScale divisions and BatchNorm affine maps)."""
    f = 0
    for lay in model.layers:
        kind = type(lay).__name__
        width = int(lay.output.shape[-1])
        if kind == "Dense":
            n_in, n_out = lay.kernel.shape
            f += 2 * int(n_in) * int(n_out) + (n_out if lay.use_bias else 0)
        elif kind in ("Multiply", "DegreeScale"):
            f += width
        elif kind == "BatchNormalization":
            f += 2 * width
    return int(f)


def m4_params(D, n_cls, K):
    """Closed form (15) of the paper: (K-1)D^2 + KCD + C + 2K."""
    return (K - 1) * D * D + K * n_cls * D + n_cls + 2 * K


def build(variant, K, extra, D, n_cls):
    from src.model import build_variant
    info = {}
    if variant in ("m4", "m4g", "m4m", "nn_mlp"):
        model = build_variant(D, n_cls, variant, max_degree=K)
    elif variant == "m2w":
        model = build_m2w(D, n_cls, extra["heads"])
    elif variant == "mlp_s":
        model, h = build_mlp_s(D, n_cls, m4_params(D, n_cls, K))
        info["hidden_width"] = h
    elif variant == "m4ms":
        model = build_m4ms(D, n_cls, K, extra["width"], extra["lam"])
    else:
        raise ValueError(variant)
    return model, info


def evaluate(model, x, y, n_cls, classes):
    from sklearn.metrics import (average_precision_score, f1_score,
                                 matthews_corrcoef)
    from sklearn.preprocessing import label_binarize
    model.predict(x[:4096], batch_size=4096, verbose=0)   # warm-up trace
    t0 = time.perf_counter()
    prob = model.predict(x, batch_size=4096, verbose=0)
    t_inf = time.perf_counter() - t0
    pred = prob.argmax(1)
    yb = label_binarize(y, classes=np.arange(n_cls))
    per = {}
    for c, nm in enumerate(classes):
        m = y == c
        if not m.any():
            continue
        tp = int(((pred == c) & m).sum())
        npred = int((pred == c).sum())
        per[nm] = {"n": int(m.sum()),
                   "recall": tp / int(m.sum()),
                   "precision": tp / npred if npred else 0.0,
                   "pr_auc": float(average_precision_score(yb[:, c],
                                                           prob[:, c]))}
    return {
        "pr_auc_macro": float(average_precision_score(yb, prob,
                                                      average="macro")),
        "pr_auc_micro": float(average_precision_score(yb, prob,
                                                      average="micro")),
        "f1_macro": float(f1_score(y, pred, average="macro",
                                   zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "per_class": per,
        "inference_s": t_inf,
        "inference_us_per_flow": 1e6 * t_inf / len(y),
    }


def train_one(stem, d, meta, variant, K, extra, seed, quiet=True):
    import pandas as pd
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras import callbacks
    from tensorflow.keras.utils import to_categorical

    name = run_name(stem, variant, K, extra, seed)
    out_json = RUN_DIR / f"{name}.json"
    if out_json.exists():
        print(f"  [skip] {name} (result exists)", flush=True)
        return
    set_seed(seed)
    x_tr, y_tr = d["x_train"], d["y_train"]
    x_va, y_va = d["x_val"], d["y_val"]
    n_cls, classes = meta["n_classes"], list(meta["label_mapping"])
    cls = np.unique(y_tr)
    cw = dict(zip(map(int, cls), compute_class_weight(
        "balanced", classes=cls, y=y_tr)))

    model, info = build(variant, K, extra, x_tr.shape[1], n_cls)
    ckpt = MODEL_DIR / f"{name}{C.MODEL_EXT}"
    cbs = [
        callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=C.EARLY_STOP_PATIENCE,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=0),
    ]
    print(f"\n===== {name}  params={model.count_params():,} =====",
          flush=True)
    t0 = time.perf_counter()
    hist = model.fit(x_tr, to_categorical(y_tr, n_cls),
                     validation_data=(x_va, to_categorical(y_va, n_cls)),
                     epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
                     class_weight=cw, callbacks=cbs,
                     verbose=0 if quiet else 2)
    t_train = time.perf_counter() - t0
    h = pd.DataFrame(hist.history)
    h.to_csv(HIST_DIR / f"{name}.csv", index=False)

    res = evaluate(model, d["x_test"], d["y_test"], n_cls, classes)
    seen = d["seen_test"] if "seen_test" in d.files else None
    if seen is not None and seen.any() and (~seen).any():
        res["unseen_only"] = {k: v for k, v in evaluate(
            model, d["x_test"][~seen], d["y_test"][~seen], n_cls,
            classes).items() if k != "inference_s"}
    res.update({
        "run": name, "dataset": meta["dataset"], "tag": meta["tag"],
        "variant": variant, "K": K, **extra, **info, "seed": seed,
        "params": int(model.count_params()),
        "flops_per_flow": flops_per_flow(model),
        "epochs_run": int(len(h)),
        "best_epoch": int(h["val_loss"].idxmin()) + 1,
        "best_val_loss": float(h["val_loss"].min()),
        "train_s": t_train,
        "train_s_per_epoch": t_train / max(len(h), 1),
        "hit_max_epochs": bool(len(h) >= C.EPOCHS),
    })
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  macro PR-AUC {res['pr_auc_macro']:.4f}  "
          f"epochs {res['epochs_run']}  train {t_train / 60:.1f} min",
          flush=True)


def parse_only(spec):
    """'m4:K4', 'm2w:K2:H3', 'm4ms:K4:W16:L0.01' -> (variant, K, extra)."""
    p = spec.split(":")
    v, K, extra = p[0], 4, {}
    for t in p[1:]:
        if t[0] == "K":
            K = int(t[1:])
        elif t[0] == "H":
            extra["heads"] = int(t[1:])
        elif t[0] == "W":
            extra["width"] = int(t[1:])
        elif t[0] == "L":
            extra["lam"] = float(t[1:])
    if v == "m4ms":
        extra.setdefault("width", 16)
        extra.setdefault("lam", 1e-3)
    return v, K, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ciciot2023", "nfunsw"],
                    required=True)
    ap.add_argument("--tag", default="p2r")
    ap.add_argument("--plan", choices=["stage1"], default=None)
    ap.add_argument("--only", nargs="+", default=None,
                    help="explicit configs, e.g. m4:K4 m2w:K2:H3")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="intra-op threads (0 = TensorFlow default)")
    a = ap.parse_args()
    set_threads(a.threads)

    stem = f"{a.dataset}_{a.tag}"
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads((C.PROCESSED_DIR / f"{stem}_meta.json")
                      .read_text("utf-8"))
    if a.only:
        configs = [parse_only(s) for s in a.only]
    elif a.plan == "stage1":
        configs = STAGE1
    else:
        ap.error("give --plan or --only")

    seeds = range(a.seed_start, a.seed_start + a.seeds)
    total = len(configs) * len(seeds)
    i = 0
    # seeds outermost: after seed 1 every configuration has one result
    for s in seeds:
        for v, K, extra in configs:
            i += 1
            print(f"[{i}/{total}]", end="", flush=True)
            train_one(stem, d, meta, v, K, extra, s)


if __name__ == "__main__":
    main()
