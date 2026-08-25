"""다중 시드 실행 — 모델 비교의 전제조건.

단일 실행에서 관측된 변동폭이 모델 간 차이보다 커서, 시드를 바꿔가며
반복하지 않으면 어떤 비교도 신뢰할 수 없다. TensorFlow CPU 연산은 시드를
고정해도 병렬 축약 순서 때문에 완전히 결정적이지 않다.

산출물
  outputs/tables/seeds_{stem}.csv     시드별 원시 지표 (기존 행과 병합)
  outputs/tables/summary_{stem}.csv   평균 +- 표준편차

실행:
  python -m src.multiseed --dataset ciciot2023 --tag full --variants m4 nn_mlp --seeds 5
  python -m src.multiseed --dataset nfunsw --tag full --variants m4m nn_mlp --seeds 5 --reuse

--reuse 는 저장된 체크포인트를 그대로 평가한다. 학습이 끝난 뒤 seeds_*.csv
를 잃어버렸을 때 재학습 없이 복구하는 용도다.
"""

import argparse
import json

import numpy as np
import pandas as pd
from tensorflow.keras import models

from src import config as C
from src.evaluate import metrics_of
from src.model import CUSTOM_OBJECTS
from src.train import run as train_run

KEYS = ["Accuracy", "Precision", "Recall", "F1-score", "MCC",
        "PR-AUC(macro)", "PR-AUC(micro)"]


def evaluate_model(path, x, y, n_classes):
    m = models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
    prob = m.predict(x, batch_size=4096, verbose=0)
    return metrics_of(y, np.argmax(prob, axis=1), prob, n_classes)


def existing_ckpt(stem, variant, seed):
    """Return the saved checkpoint for this run, or None if it is absent.

    train.py names it {stem}_{variant}.keras for the default seed and
    {stem}_{variant}_s{seed}.keras otherwise. Used by --reuse to rebuild a
    lost seeds_*.csv from checkpoints that are still on disk, without
    spending hours retraining models that already exist.
    """
    suffix = "" if seed == C.SEED else f"_s{seed}"
    p = C.MODEL_DIR / f"{stem}_{variant}{suffix}{C.MODEL_EXT}"
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variants", nargs="+", default=["m4"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--max-degree", type=int, default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="evaluate existing checkpoints instead of retraining")
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}" if a.tag else a.dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8"))
    x_te, y_te = d["x_test"], d["y_test"]
    n_classes = meta["n_classes"]

    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    rows = []
    for v in a.variants:
        for s in seeds:
            print(f"\n===== {stem} / {v} / seed={s} =====", flush=True)
            ckpt = existing_ckpt(stem, v, s) if a.reuse else None
            if ckpt is None:
                if a.reuse:
                    print("  [warn] --reuse was given but no checkpoint "
                          "exists for this run; training it now.", flush=True)
                ckpt = train_run(a.dataset, v, a.tag, seed=s, quiet=True,
                                 max_degree=a.max_degree)
            else:
                print(f"  [reuse] {ckpt.name}", flush=True)
            m = evaluate_model(ckpt, x_te, y_te, n_classes)
            m.update({"Variant": v, "seed": s})
            rows.append(m)
            print(f"  PR-AUC(macro)={m['PR-AUC(macro)']:.4f}  "
                  f"F1={m['F1-score']:.4f}  MCC={m['MCC']:.4f}", flush=True)

    raw = pd.DataFrame(rows)[["Variant", "seed"] + KEYS]
    out = C.TABLE_DIR / f"seeds_{stem}.csv"
    if out.exists():
        # Merge instead of overwrite: running one variant must not delete
        # the rows of variants measured earlier. This is how the m4m and
        # nn_mlp rows of seeds_nfunsw_full.csv were lost.
        prev = pd.read_csv(out)
        if {"Variant", "seed"} <= set(prev.columns):
            raw = pd.concat([prev, raw], ignore_index=True)
            raw = raw.drop_duplicates(subset=["Variant", "seed"], keep="last")
            raw = raw.sort_values(["Variant", "seed"]).reset_index(drop=True)
    raw.to_csv(out, index=False)

    g = raw.groupby("Variant")[KEYS]
    summ = g.mean().round(4).add_suffix("_mean").join(
        g.std().round(4).add_suffix("_std"))
    summ["n_seeds"] = g.size()
    summ = summ.reset_index().sort_values("PR-AUC(macro)_mean",
                                          ascending=False)
    summ.to_csv(C.TABLE_DIR / f"summary_{stem}.csv", index=False)

    print(f"\n{'=' * 70}\n  시드별 원시값\n{'=' * 70}")
    print(raw.round(4).to_string(index=False))

    print(f"\n{'=' * 70}\n  요약 (평균 +- 표준편차)\n{'=' * 70}")
    for _, r in summ.iterrows():
        print(f"  {r['Variant']:8s}  "
              f"PR-AUC(macro) {r['PR-AUC(macro)_mean']:.4f} "
              f"+- {r['PR-AUC(macro)_std']:.4f}   "
              f"F1 {r['F1-score_mean']:.4f} +- {r['F1-score_std']:.4f}   "
              f"MCC {r['MCC_mean']:.4f} +- {r['MCC_std']:.4f}")

    print(f"\n  쌍별 유의성 검정은 아래로 실행하세요 (재학습 불필요):")
    print(f"    python -m src.compare --dataset {a.dataset}"
          + (f" --tag {a.tag}" if a.tag else ""))


if __name__ == "__main__":
    main()
