"""1단계 Baseline.

표 형식 데이터에서 GBDT 가 신경망을 상회하는 경우가 흔하므로,
제안 모델의 위치를 먼저 확정해야 논문 프레이밍이 정해진다.

포함 모델
  logreg : Logistic Regression      1차 상호작용만 사용하는 하한선
  rf     : Random Forest            스케일 불변, 표 형식 표준
  hgb    : HistGradientBoosting     LightGBM 계열. 표 형식 최강 후보
  (신경망 MLP 는 src.train --variant nn_mlp 로 별도 학습)

실행:
  python -m src.baselines --dataset ciciot2023 --tag full
  python -m src.baselines --dataset nfunsw --tag full --models hgb rf
  python -m src.baselines --dataset nfunsw --tag full --max-train 400000
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.evaluate import metrics_of, per_class_report

MODELS = {
    # max_iter=200 에서는 다섯 스케일러 모두 lbfgs 가 수렴하지 않았다.
    # 그 상태로 비교하면 스케일러 간 차이가 모델의 표현력이 아니라
    # 최적화 문제의 조건수를 재는 것이 되므로 상한을 올린다.
    # n_jobs 는 scikit-learn 1.8부터 무효이므로 제거한다.
    "logreg": lambda seed: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed),
    "hgb": lambda seed: HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        class_weight="balanced", random_state=seed),
}

DESC = {
    "logreg": "Logistic Regression (linear, 1st order only)",
    "rf": "Random Forest (scale-invariant)",
    "hgb": "HistGradientBoosting (tabular SOTA candidate)",
}


def load(dataset, tag):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
    return d, meta, stem


def run(dataset, tag, model_names, max_train, seed):
    d, meta, stem = load(dataset, tag)
    names = C.DATASETS[dataset]["classes"]
    n_classes = meta["n_classes"]

    x_tr, y_tr = d["x_train"], d["y_train"]
    x_te, y_te = d["x_test"], d["y_test"]

    if max_train and len(y_tr) > max_train:
        rng = np.random.default_rng(seed)
        # 계층 축소: 클래스 비율을 유지한 채 표본 수만 줄인다
        keep = []
        for c in np.unique(y_tr):
            idx = np.flatnonzero(y_tr == c)
            n = max(1, int(round(len(idx) * max_train / len(y_tr))))
            keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
        sel = np.sort(np.concatenate(keep))
        x_tr, y_tr = x_tr[sel], y_tr[sel]
        print(f"[축소] train {len(sel):,}행으로 계층 축소")

    print(f"[data] train={x_tr.shape}  test={x_te.shape}  "
          f"classes={n_classes}")

    rows = []
    for name in model_names:
        print(f"\n{'=' * 58}\n  {name} : {DESC[name]}\n{'=' * 58}", flush=True)
        clf = MODELS[name](seed)
        t0 = time.time()
        clf.fit(x_tr, y_tr)
        fit_s = time.time() - t0

        prob = clf.predict_proba(x_te)
        pred = np.argmax(prob, axis=1)
        m = metrics_of(y_te, pred, prob, n_classes)
        m.update({"Variant": name, "Desc": DESC[name],
                  "fit_sec": round(fit_s, 1)})
        rows.append(m)
        print(f"  학습 {fit_s:.1f}초  PR-AUC(macro)={m['PR-AUC(macro)']:.4f}  "
              f"F1={m['F1-score']:.4f}  MCC={m['MCC']:.4f}")

        pc = per_class_report(y_te, prob, names)
        pc.insert(0, "Variant", name)
        pc.to_csv(C.TABLE_DIR / f"perclass_{stem}_{name}.csv", index=False)
        print(pc.round(4).to_string(index=False))

    cols = ["Variant", "Desc", "Accuracy", "Precision", "Recall", "F1-score",
            "MCC", "PR-AUC(macro)", "PR-AUC(micro)", "fit_sec"]
    df = pd.DataFrame(rows)[cols]
    out = C.TABLE_DIR / f"metrics_baseline_{stem}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 58}\n  Baseline 요약 ({stem})\n{'=' * 58}")
    print(df.round(4).to_string(index=False))

    # 제안 모델과 한 표에 합쳐 비교
    prop = C.TABLE_DIR / f"metrics_{stem}.csv"
    if prop.exists():
        both = pd.concat([pd.read_csv(prop), df], ignore_index=True)
        both = both.sort_values("PR-AUC(macro)", ascending=False)
        both.to_csv(C.TABLE_DIR / f"metrics_all_{stem}.csv", index=False)
        print(f"\n{'=' * 58}\n  전체 비교 (PR-AUC macro 내림차순)\n{'=' * 58}")
        print(both[["Variant", "Accuracy", "F1-score", "MCC",
                    "PR-AUC(macro)", "PR-AUC(micro)"]]
              .round(4).to_string(index=False))
    print(f"\n[save] {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=list(MODELS))
    ap.add_argument("--max-train", type=int, default=0,
                    help="학습 표본 상한. RF 가 느리면 400000 등으로 지정")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    run(a.dataset, a.tag, a.models, a.max_train, a.seed)


if __name__ == "__main__":
    main()
