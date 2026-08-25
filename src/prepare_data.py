"""원본 대용량 CSV -> raw CSV (샘플링 선택).

수행 내용
  1. inf -> nan 치환 후 결측 행 제거
  2. CICIoT2023 은 34개 세부 라벨을 8개 카테고리로 통합
  3. --max-per-class 0 이면 샘플링 없이 원본 분포 그대로 유지
  4. 청크 단위 처리 + float32 다운캐스트로 메모리 절감

원본 분포 (참고)
  CICIoT2023 : 1,176,851 행. DDoS 855,981 ... BruteForce 319 (Benign 2.4%)
  NF-UNSW-v3 : 2,242,931 행. Benign 2,151,027 ... Worms 136 (Benign 95.9%)

실행:
  python -m src.prepare_data --dataset nfunsw --max-per-class 0      # 전체
  python -m src.prepare_data --dataset nfunsw                        # 상한 1만
"""

import argparse

import numpy as np
import pandas as pd

from src import config as C

CHUNK = 200_000


def downcast(df):
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def resolve_mapping(src, col, mapping):
    """이미 카테고리로 매핑된 파일이면 매핑을 건너뛴다.

    Data_*.csv 처럼 이미 통합된 파일을 source/ 에 넣으면 매핑 결과가
    전부 NaN 이 되어 0행이 되는 사고를 막는다.
    """
    if mapping is None:
        return None
    head = pd.read_csv(src, usecols=[col], nrows=50_000)
    labels = set(head[col].dropna().unique())
    if labels & set(mapping):
        return mapping
    if labels <= set(mapping.values()):
        print(f"[map]  이미 카테고리 라벨({len(labels)}종)이라 매핑을 건너뜁니다.")
        return None
    raise ValueError(
        f"라벨을 인식할 수 없습니다: {sorted(labels)[:8]} ...\n"
        f"       원본(세부 라벨) 또는 카테고리 라벨 파일인지 확인하세요."
    )


def clean_chunk(df, col, mapping):
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if mapping is not None:
        df = df.copy()
        df[col] = df[col].map(mapping)
        df = df[df[col].notna()]
    if n0 and len(df) == 0:
        raise ValueError(
            "청크의 모든 행이 제거되었습니다. 라벨 매핑 또는 결측 처리를 "
            "확인하세요."
        )
    return downcast(df)


def run(name, max_per_class):
    spec = C.DATASETS[name]
    src = C.SOURCE_DIR / spec["raw_file"]
    if not src.exists():
        raise FileNotFoundError(f"{src} 없음. 원본 CSV 를 data/source/ 에 두세요.")

    col = spec["label_col"]
    mp = spec.get("category_map")
    mapping = resolve_mapping(src, col, getattr(C, mp) if mp else None)

    stem = spec["raw_file"].replace(".csv", "")
    suffix = "_full" if max_per_class == 0 else ""
    dst = C.RAW_DIR / f"{stem}{suffix}.csv"

    print(f"[load] {src.name}  (청크 {CHUNK:,}행 단위)")

    if max_per_class == 0:
        # 전체 사용: 청크마다 정제해서 바로 append
        total, first = 0, True
        counts = {}
        for i, ch in enumerate(pd.read_csv(src, chunksize=CHUNK,
                                           low_memory=False)):
            ch = clean_chunk(ch, col, mapping)
            ch.to_csv(dst, mode="w" if first else "a",
                      header=first, index=False)
            for k, v in ch[col].value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
            total += len(ch)
            first = False
            print(f"  chunk {i + 1}: 누적 {total:,}행", end="\r")
        print(f"\n[분포] 원본 그대로 (총 {total:,}행)")
        s = pd.Series(counts).sort_values(ascending=False)
        print((s.to_frame("count")
                .assign(pct=lambda d: (100 * d["count"] / total).round(2))
                .to_string()))
    else:
        # 상한 샘플링: 전체를 메모리에 올려야 계층 샘플링이 정확하다
        df = pd.concat(
            [clean_chunk(ch, col, mapping)
             for ch in pd.read_csv(src, chunksize=CHUNK, low_memory=False)],
            ignore_index=True,
        )
        print(f"\n[원본 분포] ({len(df):,}행)")
        print(df[col].value_counts().to_string())
        parts = [g.sample(n=max_per_class, random_state=42)
                 if len(g) >= max_per_class else g
                 for _, g in df.groupby(col, sort=True)]
        out = pd.concat(parts).reset_index(drop=True)
        print(f"\n[샘플링 후] (상한 {max_per_class:,})")
        print(out[col].value_counts().sort_index().to_string())
        out.to_csv(dst, index=False)

    print(f"\n[save] {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--max-per-class", type=int, default=C.MAX_PER_CLASS,
                    help="0 이면 샘플링 없이 원본 분포 전체 사용")
    a = ap.parse_args()
    run(a.dataset, a.max_per_class)


if __name__ == "__main__":
    main()
