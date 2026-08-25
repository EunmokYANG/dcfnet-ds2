"""학습 실행.

실행:  python -m src.train --dataset ciciot2023 --variant m3
      python -m src.train --dataset ciciot2023 --variant all
"""

import argparse
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical

from src import config as C
from src.model import build_variant


def set_seed(seed=C.SEED):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load(name):
    p = C.PROCESSED_DIR / f"{name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음. 먼저 src.preprocess 를 실행하세요.")
    d = np.load(p, allow_pickle=True)
    meta = json.loads((C.PROCESSED_DIR / f"{name}_meta.json").read_text("utf-8"))
    return d, meta


def run(dataset, variant, tag="", seed=C.SEED, quiet=False,
        max_degree=None):
    set_seed(seed)
    stem = f"{dataset}_{tag}" if tag else dataset
    d, meta = load(stem)
    x_train_all, y_train_all = d["x_train"], d["y_train"]
    n_classes = meta["n_classes"]

    # train 내부에서 validation 분리 (test 는 절대 사용하지 않음)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_all, y_train_all, test_size=C.VAL_SIZE,
        random_state=C.SEED, stratify=y_train_all,
    )

    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights = dict(zip(map(int, classes), cw))

    y_tr_oh = to_categorical(y_tr, num_classes=n_classes)
    y_val_oh = to_categorical(y_val, num_classes=n_classes)

    cfg = C.VARIANTS[variant]
    model = build_variant(x_tr.shape[1], n_classes, variant, max_degree)

    name = f"{stem}_{variant}" + (f"_s{seed}" if seed != C.SEED else "")
    ckpt = C.MODEL_DIR / f"{name}{C.MODEL_EXT}"
    cbs = [
        callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=C.EARLY_STOP_PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=1),
    ]

    print(f"\n===== {name} :: {cfg['desc']} =====")
    print(f"train={x_tr.shape}  val={x_val.shape}  classes={n_classes}")

    hist = model.fit(
        x_tr, y_tr_oh,
        validation_data=(x_val, y_val_oh),
        epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=0 if quiet else 2,
    )

    pd.DataFrame(hist.history).to_csv(
        C.TABLE_DIR / f"history_{name}.csv", index=False
    )
    print(f"[save] {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variant", choices=list(C.VARIANTS) + ["all"],
                    default="m3")
    ap.add_argument("--tag", default="", help="전처리 tag (예: m10)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--max-degree", type=int, default=None,
                    help="config.MAX_DEGREE 대신 사용할 차수 상한")
    a = ap.parse_args()
    vs = list(C.VARIANTS) if a.variant == "all" else [a.variant]
    for v in vs:
        run(a.dataset, v, a.tag, a.seed, max_degree=a.max_degree)


if __name__ == "__main__":
    main()
