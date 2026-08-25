"""섭동 검증 — 추출된 피처 조합이 실제로 해당 공격을 결정하는가.

계수가 크다는 것만으로는 인과를 주장할 수 없다. 상위 조합에 속한 피처를
test set 에서 셔플했을 때 해당 공격의 재현율만 선택적으로 떨어져야 한다.

무작위로 뽑은 같은 개수의 피처를 셔플한 대조군과 비교하여, 하락폭이
유의하게 큰지 검정한다.

실행:
  python -m src.perturbation --dataset ciciot2023 --tag full --variant m4
  python -m src.perturbation --dataset ciciot2023 --tag full --n-feat 4 --n-control 30
"""

import argparse
import json

import numpy as np
import pandas as pd
from tensorflow.keras import models

from src import config as C
from src.model import CUSTOM_OBJECTS


def _raw_score(df, attack, min_degree):
    """해당 공격의 상위 조합에서 피처별 계수 합. 클래스 내부에서 정규화."""
    sub = df[(df.Attack == attack) & (df.degree >= min_degree)]
    sc = {}
    for _, r in sub.iterrows():
        for f in str(r["features"]).replace("^2", "").split(" x "):
            f = f.strip()
            sc[f] = sc.get(f, 0.0) + r["abs_coef"]
    tot = sum(sc.values()) or 1.0
    return {k: v / tot for k, v in sc.items()}


def top_features(df, attack, n_feat, mode="specific", min_degree=2):
    """공격별 핵심 피처를 뽑는다.

    mode="raw"      : 계수 합 상위. 조합에 자주 등장하는 피처가 클래스와
                      무관하게 뽑혀, 여러 클래스가 같은 피처를 지목한다.
    mode="specific" : 다른 클래스 평균을 뺀 상대 특이성 상위. 그 공격에만
                      두드러진 피처가 남으므로 섭동의 선택성 검정에 적합.
    """
    attacks = [a for a in df.Attack.unique()]
    self_sc = _raw_score(df, attack, min_degree)
    if mode == "raw":
        return sorted(self_sc, key=self_sc.get, reverse=True)[:n_feat]

    others = [_raw_score(df, a, min_degree) for a in attacks if a != attack]
    spec = {}
    for f, v in self_sc.items():
        mean_other = np.mean([o.get(f, 0.0) for o in others]) if others else 0.0
        spec[f] = v - mean_other
    return sorted(spec, key=spec.get, reverse=True)[:n_feat]


def recall_after_shuffle(model, x, y, cols, classes, rng):
    """지정 열을 셔플한 뒤 클래스별 재현율을 잰다."""
    xs = x.copy()
    for c in cols:
        xs[:, c] = xs[rng.permutation(len(xs)), c]
    pred = np.argmax(model.predict(xs, batch_size=8192, verbose=0), axis=1)
    return np.array([(pred[y == k] == k).mean() if (y == k).sum() else np.nan
                     for k in range(len(classes))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variant", default="m4")
    ap.add_argument("--n-feat", type=int, default=3,
                    help="공격별로 셔플할 피처 수")
    ap.add_argument("--n-control", type=int, default=20,
                    help="무작위 대조군 반복 수")
    ap.add_argument("--max-test", type=int, default=200_000,
                    help="평가 표본 상한 (속도)")
    ap.add_argument("--mode", choices=["specific", "raw"], default="specific",
                    help="specific=상대 특이성 상위, raw=계수 합 상위")
    ap.add_argument("--min-degree", type=int, default=2,
                    help="피처 점수에 포함할 최소 차수")
    ap.add_argument("--suffix", default="", help="출력 파일명 접미사")
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}" if a.tag else a.dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    feats = [str(f) for f in d["feature_names"]]
    classes = C.DATASETS[a.dataset]["classes"]
    model = models.load_model(
        C.MODEL_DIR / f"{stem}_{a.variant}{C.MODEL_EXT}",
        custom_objects=CUSTOM_OBJECTS, compile=False)

    ipath = C.TABLE_DIR / f"interactions_{stem}_{a.variant}.csv"
    if not ipath.exists():
        raise FileNotFoundError(
            f"{ipath.name} 없음. 먼저 src.interactions 를 실행하세요.")
    inter = pd.read_csv(ipath)

    x, y = d["x_test"], d["y_test"]
    rng = np.random.default_rng(C.SEED)
    if len(y) > a.max_test:
        # 계층 축소
        keep = []
        for k in np.unique(y):
            idx = np.flatnonzero(y == k)
            n = max(1, int(round(len(idx) * a.max_test / len(y))))
            keep.append(rng.choice(idx, min(n, len(idx)), replace=False))
        sel = np.sort(np.concatenate(keep))
        x, y = x[sel], y[sel]
        print(f"[축소] test {len(y):,}행")

    if a.n_control < 20:
        print(f"[경고] 대조군 {a.n_control}개로는 최소 p값이 "
              f"{1 / a.n_control:.3f} 이라 p<0.05 판정이 불가능합니다. "
              f"--n-control 20 이상을 쓰세요.")

    base = recall_after_shuffle(model, x, y, [], classes, rng)
    print(f"[base] 원본 재현율 계산 완료  (mode={a.mode}, "
          f"n_feat={a.n_feat}, n_control={a.n_control})")

    fidx = {f: i for i, f in enumerate(feats)}
    rows = []
    for c, cn in enumerate(classes):
        picks = top_features(inter, cn, a.n_feat, a.mode, a.min_degree)
        cols = [fidx[f] for f in picks if f in fidx]
        if not cols:
            continue

        r = recall_after_shuffle(model, x, y, cols, classes, rng)
        drop_self = base[c] - r[c]
        drop_other = np.nanmean(np.delete(base - r, c))

        # 대조군: 같은 개수의 무작위 피처
        ctrl = []
        for _ in range(a.n_control):
            rc = rng.choice(len(feats), len(cols), replace=False)
            rr = recall_after_shuffle(model, x, y, list(rc), classes, rng)
            ctrl.append(base[c] - rr[c])
        ctrl = np.array(ctrl)
        z = ((drop_self - ctrl.mean()) / ctrl.std()
             if ctrl.std() > 0 else np.nan)
        pct = float((ctrl >= drop_self).mean())

        rows.append({
            "Attack": cn, "N": int((y == c).sum()),
            "features": ", ".join(picks),
            "recall_base": float(base[c]),
            "recall_shuffled": float(r[c]),
            "drop_self": float(drop_self),
            "drop_other_mean": float(drop_other),
            "selectivity": float(drop_self - drop_other),
            "ctrl_drop_mean": float(ctrl.mean()),
            "ctrl_drop_std": float(ctrl.std()),
            "z_vs_control": float(z),
            "p_empirical": pct,
            "causal": bool(pct < 0.05 and drop_self > drop_other),
        })
        print(f"  {cn:<16} drop {drop_self:+.4f} (대조 {ctrl.mean():+.4f}"
              f" ± {ctrl.std():.4f})  z={z:+.2f}  p={pct:.3f}")

    df = pd.DataFrame(rows)
    out = C.TABLE_DIR / f"perturb_{stem}_{a.variant}{a.suffix}.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 78}\n  섭동 검증 결과\n{'=' * 78}")
    cols = ["Attack", "N", "recall_base", "recall_shuffled", "drop_self",
            "drop_other_mean", "selectivity", "z_vs_control",
            "p_empirical", "causal"]
    print(df[cols].round(4).to_string(index=False))

    # 선택된 피처가 클래스마다 다른지 점검
    allf = [set(str(r).split(", ")) for r in df["features"]]
    uniq = len(set().union(*allf)) if allf else 0
    dup = sum(1 for i, A in enumerate(allf)
              for B in allf[i + 1:] if A & B)
    print(f"\n  선택 피처 다양성: 서로 다른 피처 {uniq}개, "
          f"겹치는 클래스 쌍 {dup}개")
    if dup > len(allf):
        print("  -> 여러 클래스가 같은 피처를 지목합니다. "
              "--mode specific 을 쓰거나 --n-feat 을 늘리세요.")

    ok = df["causal"].sum()
    print(f"\n  인과 확인: {ok}/{len(df)} 클래스 "
          f"(p<0.05 이고 자기 하락 > 타 클래스 평균 하락)")
    if ok >= len(df) * 0.5:
        print("  -> 추출된 조합이 해당 공격을 선택적으로 결정합니다.")
    else:
        print("  -> 선택성이 약합니다. 조합 해석 주장을 재검토하세요.")
    print(f"\n  [save] {out.name}")


if __name__ == "__main__":
    main()
