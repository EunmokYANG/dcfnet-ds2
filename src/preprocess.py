"""전처리: raw CSV -> train/test 분리 npz

선행 연구 대비 변경점
  1. train/test 를 먼저 분리한 뒤 StandardScaler 를 train 에만 fit 한다.
  2. 상수 컬럼 판정도 train 기준으로 수행한다.
  3. NF-UNSW-NB15-v3 의 네트워크 식별자(IP, 포트)를 제거한다.
  4. 소수 클래스 증폭(--amplify) 옵션.
     Yang, Joshi & Seo (2021), CMC 66(2), 1647-1663 의 방법을 따른다.
       n_new = n + m * n      (m = 5, 10, ..., 100)
     반드시 train/test 분할 "이후" train 에만 적용한다.
     분할 전에 복제하면 동일 행이 양쪽에 들어가 완전 누수가 된다.

실행:
  python -m src.preprocess --dataset ciciot2023
  python -m src.preprocess --dataset ciciot2023 --amplify 10 --tag m10
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, RobustScaler,
                                   StandardScaler)

from src import config as C

_T0 = [time.time()]


def step(msg):
    """단계별 경과시간과 함께 진행 상황을 출력한다."""
    now = time.time()
    print(f"  [{now - _T0[0]:6.1f}s] {msg}", flush=True)
    _T0[0] = now


def split_categorical(X_df, spec, mode):
    """명목형 코드와 비트마스크를 분리한다.

    mode = "keep"   기존 동작. 모든 컬럼을 연속형으로 다룬다.
    mode = "drop"   해당 컬럼을 제거한다.
    mode = "encode" 명목형은 원-핫, 비트마스크는 비트별 이진 컬럼으로 바꾸고
                    스케일러 대상에서 제외한다.

    반환: (스케일 대상 DataFrame, 통과 DataFrame)
    """
    nom = [c for c in spec.get("nominal_cols", []) if c in X_df.columns]
    bit = [c for c in spec.get("bitmask_cols", []) if c in X_df.columns]
    if mode == "keep" or not (nom or bit):
        return X_df, None
    num = X_df.drop(columns=nom + bit)
    if mode == "drop":
        print(f"[cat] 명목형/비트마스크 {len(nom) + len(bit)}개 제거: "
              f"{nom + bit}")
        return num, None

    parts = []
    for c in nom:
        oh = pd.get_dummies(X_df[c].astype("int64"), prefix=c)
        parts.append(oh.astype("float32"))
    for c in bit:
        v = X_df[c].astype("int64").values
        for b in range(8):
            parts.append(pd.Series(((v >> b) & 1).astype("float32"),
                                   name=f"{c}_b{b}", index=X_df.index))
    passthru = pd.concat(parts, axis=1)
    print(f"[cat] 명목형 {len(nom)}개 -> 원-핫, 비트마스크 {len(bit)}개 -> "
          f"비트 {8 * len(bit)}개. 통과 컬럼 {passthru.shape[1]}개")
    return num, passthru


def apply_scaler(kind, x_tr, x_te):
    """스케일러 적용.

    signedlog / none 은 학습 데이터에서 추정하는 파라미터가 없으므로
    배포 후 분포가 이동해도 재적합이 필요하지 않다.
    signedlog_minmax 는 비선형 압축 뒤에 아핀 경계화를 덧붙인 것이다.
    아핀 변환은 척도 불변 형태 통계량을 바꾸지 못하므로 피처 내 해상도는
    signedlog 와 같고, 출력 범위와 피처 간 균형만 개선된다.
    다만 min-max 부분이 파라미터를 적합하므로 재적합 불필요라는 이점은 잃는다.
    """
    if kind == "none":
        return x_tr, x_te
    if kind in ("signedlog", "signedlog_minmax"):
        f = lambda a: np.sign(a) * np.log1p(np.abs(a))
        x_tr = f(x_tr).astype("float32")
        x_te = f(x_te).astype("float32")
        if kind == "signedlog":
            return x_tr, x_te
        sc = MinMaxScaler()
        sc.fit(x_tr)
        return (sc.transform(x_tr).astype("float32"),
                sc.transform(x_te).astype("float32"))
    if kind == "asinh":
        # log 계열이지만 큰 |x| 에서 log(2x) 로 수렴. signedlog 의 자연스러운 대안.
        f = lambda a: np.arcsinh(a)
        return f(x_tr).astype("float32"), f(x_te).astype("float32")
    if kind == "winsor_minmax":
        # 학습 분위 1~99 로 클리핑한 뒤 min-max.
        # 클리핑이 비선형이므로 아핀 불변 명제의 적용 대상이 아니다.
        lo = np.percentile(x_tr, 1, axis=0)
        hi = np.percentile(x_tr, 99, axis=0)
        c_tr = np.clip(x_tr, lo, hi)
        c_te = np.clip(x_te, lo, hi)
        sc = MinMaxScaler()
        sc.fit(c_tr)
        return (sc.transform(c_tr).astype("float32"),
                sc.transform(c_te).astype("float32"))
    if kind == "robust_safe":
        # IQR=0 인 피처에 스케일 1 을 주어 RobustScaler 를 사용 가능하게 한다.
        med = np.median(x_tr, axis=0)
        q1, q3 = np.percentile(x_tr, [25, 75], axis=0)
        iqr = np.where((q3 - q1) > 0, q3 - q1, 1.0)
        f = lambda a: (a - med) / iqr
        return f(x_tr).astype("float32"), f(x_te).astype("float32")
    if kind == "yeojohnson":
        from sklearn.preprocessing import PowerTransformer
        sc = PowerTransformer(method="yeo-johnson", standardize=True)
        sc.fit(x_tr)
        return (sc.transform(x_tr).astype("float32"),
                sc.transform(x_te).astype("float32"))
    if kind == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        sc = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=1000, subsample=200_000,
                                 random_state=C.SEED)
        sc.fit(x_tr)
        return (sc.transform(x_tr).astype("float32"),
                sc.transform(x_te).astype("float32"))
    sc = {"standard": StandardScaler,
          "robust": RobustScaler,
          "minmax": MinMaxScaler}[kind]()
    sc.fit(x_tr)
    return (sc.transform(x_tr).astype("float32"),
            sc.transform(x_te).astype("float32"))


def make_split(name, spec, df, y, split):
    """random = 계층 무작위, temporal = 앞 기간 학습 / 뒤 기간 평가."""
    idx = np.arange(len(y))
    if split == "random":
        return train_test_split(idx, test_size=C.TEST_SIZE,
                                random_state=C.SEED, stratify=y)
    tcol = spec.get("time_col")
    if not tcol or tcol not in df.columns:
        raise ValueError(
            f"{name} 에는 타임스탬프 컬럼이 없어 시간 분할을 할 수 없습니다.")
    order = np.argsort(df[tcol].values, kind="stable")
    cut = int(len(order) * (1 - C.TEST_SIZE))
    tr, te = order[:cut], order[cut:]
    print(f"[split] 시간 분할: {tcol} 기준 앞 {len(tr):,} / 뒤 {len(te):,}")
    return tr, te


def load_raw(name, full=False):
    spec = C.DATASETS[name]
    fname = spec["raw_file"]
    if full:
        fname = fname.replace(".csv", "_full.csv")
    path = C.RAW_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. data/raw/ 에 CSV 를 두거나 "
            f"prepare_data.py 를 먼저 실행하세요."
        )
    mb = path.stat().st_size / 1024 / 1024
    print(f"[load] {path.name}  ({mb:,.0f} MB)", flush=True)
    _T0[0] = time.time()

    parts, n = [], 0
    for i, ch in enumerate(pd.read_csv(path, chunksize=300_000,
                                       low_memory=False)):
        for c in ch.select_dtypes(include=["float64"]).columns:
            ch[c] = ch[c].astype("float32")
        parts.append(ch)
        n += len(ch)
        step(f"읽는 중 {n:,}행")
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    del parts
    step(f"읽기 완료 shape={df.shape}")
    return df, spec


def amplify_minority(x, y, m, threshold, seed=C.SEED):
    """소수 클래스를 m 배 증폭. n_new = n + m*n (2021 CMC 논문 방식).

    train 에만 적용되어야 한다. 호출부에서 이를 보장한다.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [x], [y]
    report = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = len(idx)
        if n >= threshold:
            report.append((int(c), n, n, 1))
            continue
        rep = rng.choice(idx, size=m * n, replace=True)
        xs.append(x[rep])
        ys.append(y[rep])
        report.append((int(c), n, n + m * n, 1 + m))

    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    order = rng.permutation(len(y_out))

    print(f"\n[amplify] m={m}, threshold={threshold} (train 전용)")
    print("  class  before  ->   after   x")
    for c, b, a, f in report:
        mark = "  <-" if f > 1 else ""
        print(f"  {c:5d}  {b:6d}  ->  {a:6d}  x{f}{mark}")
    return x_out[order], y_out[order], report


def build(name, amplify=0, amplify_threshold=1000, tag="",
          scaler=None, split=None, full=False, categorical="keep"):
    scaler = scaler or C.SCALER
    split = split or C.SPLIT
    df, spec = load_raw(name, full)

    df_time = df.copy() if split == "temporal" else df
    drop = [c for c in spec["drop_cols"] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        print(f"[drop] 식별자/불필요 컬럼 {len(drop)}개 제거: {drop}")

    label_col = spec["label_col"]
    y_raw = df[label_col].values
    X_df = df.drop(columns=[label_col])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    step("라벨 인코딩 완료")
    mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
    print(f"[label] {len(mapping)} classes -> {mapping}")

    idx_tr, idx_te = make_split(name, spec, df_time, y, split)
    step(f"분할 완료 train={len(idx_tr):,} test={len(idx_te):,}")

    # 상수 컬럼 판정을 train 에만 근거해 수행
    keep = X_df.columns[X_df.iloc[idx_tr].nunique() > 1].tolist()
    removed = [c for c in X_df.columns if c not in keep]
    if removed:
        print(f"[const] 상수 컬럼 {len(removed)}개 제거: {removed}")
    X_df = X_df[keep]

    step("상수 컬럼 판정 완료")
    X_num, X_pass = split_categorical(X_df, spec, categorical)
    keep = list(X_num.columns) + (list(X_pass.columns) if X_pass is not None
                                  else [])
    X = X_num.values.astype("float32")
    step(f"배열 변환 완료 {X.shape}")
    x_train, x_test = X[idx_tr], X[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]

    x_train, x_test = apply_scaler(scaler, x_train, x_test)
    step(f"스케일링 완료 ({scaler})")

    if X_pass is not None:
        P = X_pass.values.astype("float32")
        x_train = np.hstack([x_train, P[idx_tr]])
        x_test = np.hstack([x_test, P[idx_te]])
        step(f"통과 컬럼 결합 완료 {x_train.shape}")

    # 배포 안정성 지표: 테스트 값이 학습 시 관측 범위를 벗어나는 정도
    tr_lo, tr_hi = x_train.min(axis=0), x_train.max(axis=0)
    oob = ((x_test < tr_lo) | (x_test > tr_hi))
    absmax_tr = float(np.abs(x_train).max())
    absmax_te = float(np.abs(x_test).max())
    exceed = absmax_te / (absmax_tr + 1e-12)
    n_oob_rows = int(oob.any(1).sum())
    n_oob_vals = int(oob.sum())

    # 피처별 상대 초과폭: 학습 범위 폭을 1 로 볼 때 테스트가 얼마나 더 나가는가.
    # 전체 절대최대값 비교(exceed)는 스케일이 큰 피처 하나에 지배되어
    # 다른 피처의 범위 이탈을 감추므로 이 지표를 주 진단으로 쓴다.
    width = tr_hi.astype("float64") - tr_lo.astype("float64")
    over = np.maximum(x_test.max(axis=0).astype("float64") - tr_hi.astype("float64"),
                      tr_lo.astype("float64") - x_test.min(axis=0).astype("float64"))
    over = np.maximum(over, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(width > 0, over / width, 0.0)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    j_worst = int(np.argmax(rel))
    exceed_rel = 1.0 + float(rel[j_worst])  # 1.0 이면 학습 범위 내
    worst_feat = keep[j_worst]

    print(f"[scale] {scaler}  |x|max train={absmax_tr:.4f} "
          f"test={absmax_te:.4f}")
    print(f"        범위 초과: 값 {n_oob_vals}개 ({100 * oob.mean():.5f}%) / "
          f"행 {n_oob_rows}개 ({100 * oob.any(1).mean():.5f}%)")
    print(f"        상대 초과배율 max={exceed_rel:.4f}x "
          f"(피처 '{worst_feat}') / 4차 외삽={exceed_rel ** 4:.2f}x")
    with np.errstate(over="ignore", invalid="ignore"):
        xd = x_train.astype("float64")
        rms = []
        for k in range(1, 5):
            v = np.sqrt(np.mean(xd ** (2 * k)))
            rms.append(float(v) if np.isfinite(v) else float("inf"))
        ratio = (rms[3] / rms[0]) if np.isfinite(rms[3]) and rms[0] else float("inf")
    print("        rms(x^k) = " +
          "  ".join(f"k{k + 1}:{r:.4g}" for k, r in enumerate(rms)))
    print(f"        동적범위 k4/k1 = {ratio:.4g}   "
          f"(1 에 가까울수록 차수 간 균형)")

    # 스케일러를 가르는 두 축을 실측한다.
    #  (1) 피처 간 균형: 정규화가 스칼라 하나로 이뤄지므로 피처별 rms 격차가
    #      그대로 남는다. 격차가 크면 작은 피처가 소멸한다.
    #  (2) 피처 내 해상도: IQR/전범위. 아핀 변환에는 불변이고
    #      비선형 압축(signed-log)에서만 커진다.
    xd64 = x_train.astype("float64")
    f_rms = np.sqrt(np.mean(xd64 ** 2, axis=0))
    pos = f_rms[f_rms > 0]
    feat_rms_ratio = float(pos.max() / pos.min()) if len(pos) else float("inf")
    n_vanish = int((f_rms / (f_rms.max() + 1e-30) < 1e-4).sum())
    q1, q3 = np.percentile(xd64, [25, 75], axis=0)
    span = xd64.max(axis=0) - xd64.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iqr_ratio = np.where(span > 0, (q3 - q1) / span, 0.0)
    iqr_ratio = np.nan_to_num(iqr_ratio)
    iqr_over_range = float(np.median(iqr_ratio))

    # 이진 피처는 Q1=Q3 이라 IQR=0 이 되어 중앙값을 0 으로 끌어내린다.
    # 고유값 3개 이상인 연속형 피처만 따로 집계하고, 배열도 함께 남겨
    # 요약 통계를 나중에 바꿔도 재실행이 필요 없게 한다.
    n_uniq = np.array([len(np.unique(xd64[:, j]))
                       for j in range(xd64.shape[1])])
    cont = n_uniq > 2
    iqr_cont = float(np.median(iqr_ratio[cont])) if cont.any() else 0.0
    print(f"        피처간 rms 최대/최소 = {feat_rms_ratio:.4g}  "
          f"(소멸 피처 {n_vanish}/{len(f_rms)})")
    print(f"        피처내 해상도 IQR/전범위: 전체 중앙값 {iqr_over_range:.4g} / "
          f"연속형({int(cont.sum())}개) 중앙값 {iqr_cont:.4g}")

    amp_report = None
    if amplify > 0:
        x_train, y_train, amp_report = amplify_minority(
            x_train, y_train, amplify, amplify_threshold
        )

    stem = f"{name}_{tag}" if tag else name
    out = C.PROCESSED_DIR / f"{stem}.npz"
    np.savez(
        out,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        feature_names=np.array(keep, dtype=object),
    )

    meta = {
        "dataset": name,
        "tag": tag,
        "amplify_m": amplify,
        "amplify_threshold": amplify_threshold if amplify else None,
        "amplify_report": amp_report,
        "n_features": len(keep),
        "n_classes": int(len(mapping)),
        "label_mapping": mapping,
        "feature_names": keep,
        "removed_constant": removed,
        "removed_identifier": drop,
        "scaler": scaler,
        "oob_value_pct": float(100 * oob.mean()),
        "oob_row_pct": float(100 * oob.any(1).mean()),
        "oob_value_count": n_oob_vals,
        "oob_row_count": n_oob_rows,
        "absmax_train": absmax_tr,
        "absmax_test": absmax_te,
        "exceed_ratio": float(exceed),
        "exceed_ratio_deg4": float(exceed ** 4),
        "exceed_rel": exceed_rel,
        "exceed_rel_deg4": float(exceed_rel ** 4),
        "exceed_worst_feature": worst_feat,
        "feat_rms_ratio": feat_rms_ratio,
        "n_vanishing_features": n_vanish,
        "iqr_over_range_median": iqr_over_range,
        "iqr_over_range_median_continuous": iqr_cont,
        "n_continuous_features": int(cont.sum()),
        "feature_n_unique": n_uniq.tolist(),
        "feature_rms": f_rms.tolist(),
        "feature_iqr_over_range": iqr_ratio.tolist(),
        "rms_u": rms,
        "dynamic_range_u4_u1": ratio,
        "split": split,
        "full": full,
        "class_counts_train": {int(k): int(v) for k, v in
                               zip(*np.unique(y_train, return_counts=True))},
        "class_counts_test": {int(k): int(v) for k, v in
                              zip(*np.unique(y_test, return_counts=True))},
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
    }
    (C.PROCESSED_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    step(f"저장 완료 {out.name} ({out.stat().st_size / 1024 / 1024:,.0f} MB)")
    print(f"[save] {out.name}  train={x_train.shape}  test={x_test.shape}",
          flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--amplify", type=int, default=0,
                    help="소수 클래스 증폭 배수 m (0=사용 안 함)")
    ap.add_argument("--amplify-threshold", type=int, default=1000,
                    help="이 개수 미만인 클래스를 증폭 대상으로 삼는다")
    ap.add_argument("--tag", default="", help="출력 파일명 접미사 (예: m10)")
    ap.add_argument("--scaler", choices=["minmax", "standard", "robust",
                                         "robust_safe", "signedlog",
                                         "signedlog_minmax", "asinh",
                                         "winsor_minmax", "yeojohnson",
                                         "quantile", "none"],
                    default=None)
    ap.add_argument("--split", choices=["random", "temporal"], default=None)
    ap.add_argument("--full", action="store_true",
                    help="원본 전체(_full.csv) 사용")
    ap.add_argument("--categorical", choices=["keep", "drop", "encode"],
                    default="keep",
                    help="명목형 코드/비트마스크 처리 방식")
    args = ap.parse_args()
    names = list(C.DATASETS) if args.dataset == "all" else [args.dataset]
    for n in names:
        print(f"\n===== {n} =====")
        build(n, args.amplify, args.amplify_threshold, args.tag,
              args.scaler, args.split, args.full, args.categorical)


if __name__ == "__main__":
    main()
