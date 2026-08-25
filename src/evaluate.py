"""평가: 지표 표, 혼동행렬, ROC/PR 곡선.

주의: 축 범위와 눈금 라벨을 항상 일치시킨다. 확대가 필요하면
      제목에 확대 구간을 명시하고 라벨은 실제 값을 그대로 쓴다.

실행:  python -m src.evaluate --dataset ciciot2023 --variants m0 m3
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)
from sklearn.preprocessing import label_binarize
from tensorflow.keras import models

from src import config as C
from src.model import CUSTOM_OBJECTS


def metrics_of(y_true, y_pred, prob=None, n_classes=None):
    """극단 불균형에서는 accuracy 가 무의미하므로 PR-AUC 를 주 지표로 둔다."""
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro",
                             zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if prob is not None:
        yb = label_binarize(y_true, classes=np.arange(n_classes))
        m["PR-AUC(macro)"] = average_precision_score(yb, prob, average="macro")
        m["PR-AUC(micro)"] = average_precision_score(yb, prob, average="micro")
    return m


def per_class_report(y_true, prob, class_names):
    """클래스별 재현율과 PR-AUC. 2021 CMC 논문의 유형별 탐지 관점."""
    n = len(class_names)
    pred = np.argmax(prob, axis=1)
    yb = label_binarize(y_true, classes=np.arange(n))
    rows = []
    for c, nm in enumerate(class_names):
        m = y_true == c
        if m.sum() == 0:
            continue
        rows.append({
            "Attack": nm, "N": int(m.sum()),
            "Recall": float((pred[m] == c).mean()),
            "Precision": float((y_true[pred == c] == c).mean())
            if (pred == c).sum() else 0.0,
            "PR-AUC": float(average_precision_score(yb[:, c], prob[:, c])),
        })
    return pd.DataFrame(rows)


def evaluate(dataset, variants, tag=""):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8")
    )
    names = C.DATASETS[dataset]["classes"]
    x_te, y_te = d["x_test"], d["y_test"]
    n_classes = meta["n_classes"]

    rows, probs = [], {}
    for v in variants:
        p = C.MODEL_DIR / f"{stem}_{v}{C.MODEL_EXT}"
        if not p.exists():
            print(f"[skip] {p.name} 없음")
            continue
        model = models.load_model(p, custom_objects=CUSTOM_OBJECTS,
                                  compile=False)
        pr = model.predict(x_te, batch_size=1024, verbose=0)
        probs[v] = pr
        rows.append({"Variant": v, "Desc": C.VARIANTS[v]["desc"],
                     **metrics_of(y_te, np.argmax(pr, axis=1), pr, n_classes)})

        pc = per_class_report(y_te, pr, names)
        pc.insert(0, "Variant", v)
        pc.to_csv(C.TABLE_DIR / f"perclass_{stem}_{v}.csv", index=False)
        print(f"\n[클래스별] {v}")
        print(pc.round(4).to_string(index=False))

        cm = confusion_matrix(y_te, np.argmax(pr, axis=1))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    square=True, ax=ax,
                    xticklabels=names, yticklabels=names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix — {stem} / {v}")
        fig.tight_layout()
        fig.savefig(C.FIG_DIR / f"cm_{stem}_{v}.png", dpi=300)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLE_DIR / f"metrics_{stem}.csv", index=False)
    print(df.to_string(index=False))

    y_bin = label_binarize(y_te, classes=np.arange(n_classes))

    # ROC — 축 범위와 라벨 일치
    fig, ax = plt.subplots(figsize=(5, 4))
    for v, pr in probs.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), pr.ravel())
        ax.plot(fpr, tpr, lw=2, label=f"{v} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve (micro-average) — {stem}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"roc_{stem}.png", dpi=300)
    plt.close(fig)

    # PR — 축 범위와 라벨 일치
    fig, ax = plt.subplots(figsize=(5, 4))
    for v, pr in probs.items():
        prec, rec, _ = precision_recall_curve(y_bin.ravel(), pr.ravel())
        ap = average_precision_score(y_bin, pr, average="micro")
        ax.plot(rec, prec, lw=2, label=f"{v} (AP = {ap:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve (micro-average) — {stem}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"pr_{stem}.png", dpi=300)
    plt.close(fig)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variants", nargs="+", default=list(C.VARIANTS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    evaluate(a.dataset, a.variants, a.tag)


if __name__ == "__main__":
    main()
