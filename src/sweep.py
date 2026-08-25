"""조건 스윕 자동화.

실험 조건(차수 상한, 스케일러, 분할, 불균형 처리)을 바꿔가며 학습·평가하고
결과를 한 표로 모은다. config.py 를 고칠 필요가 없다.

지원하는 스윕
  degree   차수 상한 K = 2,3,4,5,6
  scaler   minmax / standard / robust / signedlog / none
  split    random / temporal
  amplify  소수 클래스 증폭 배수 (2021 CMC 논문 방식)

실행 예)
  python -m src.sweep --kind degree --values 2 3 4 5 6
  python -m src.sweep --kind scaler --values minmax standard signedlog none
  python -m src.sweep --kind degree --values 2 3 4 --seeds 3
  python -m src.sweep --kind scaler --values minmax signedlog --dataset nfunsw
"""

import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src import config as C

KINDS = ["degree", "scaler", "split", "amplify"]


def sh(args, label, quiet=True):
    cmd = [sys.executable] + args
    if not quiet:
        print(f"  $ {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if r.returncode != 0:
        print(f"  [실패] {label}")
        if quiet and r.stdout:
            print(r.stdout[-1500:])
        return False
    return True


def prep_needed(kind):
    """전처리를 조건마다 다시 해야 하는 스윕인가."""
    return kind in ("scaler", "split", "amplify")


def tag_for(kind, val, base_tag):
    return f"{base_tag}_{kind}{val}" if base_tag else f"{kind}{val}"


def run_one(dataset, kind, val, a, base_tag):
    """조건 하나를 처리하고 지표 dict 를 반환한다."""
    tag = tag_for(kind, val, base_tag)
    ds = ["--dataset", dataset]
    t0 = time.time()

    # --- 전처리 (필요한 조건만)
    if prep_needed(kind):
        pre = ["-m", "src.preprocess", *ds, "--tag", tag]
        if a.full:
            pre.append("--full")
        pre += ["--scaler", val if kind == "scaler" else a.scaler]
        split = val if kind == "split" else a.split
        if split == "temporal" and not C.DATASETS[dataset].get("time_col"):
            split = "random"
        pre += ["--split", split]
        if kind == "amplify" and int(val) > 0:
            pre += ["--amplify", str(val)]
        if not sh(pre, f"preprocess {tag}"):
            return None
    else:
        # 기존 npz 를 그대로 쓴다 (차수 스윕)
        src = C.PROCESSED_DIR / f"{dataset}_{base_tag}.npz" if base_tag \
            else C.PROCESSED_DIR / f"{dataset}.npz"
        dst = C.PROCESSED_DIR / f"{dataset}_{tag}.npz"
        if not dst.exists():
            if not src.exists():
                print(f"  [건너뜀] {src.name} 없음")
                return None
            import shutil
            shutil.copy(src, dst)
            shutil.copy(src.with_name(src.stem + "_meta.json"),
                        dst.with_name(dst.stem + "_meta.json"))

    # --- 학습 + 평가 (다중 시드)
    ms = ["-m", "src.multiseed", *ds, "--tag", tag,
          "--variants", *a.variants, "--seeds", str(a.seeds)]
    if kind == "degree":
        ms += ["--max-degree", str(val)]
    if not sh(ms, f"multiseed {tag}"):
        return None

    # --- 결과 수집
    seeds = C.TABLE_DIR / f"seeds_{dataset}_{tag}.csv"
    if not seeds.exists():
        return None
    df = pd.read_csv(seeds)
    rows = []
    for v, g in df.groupby("Variant"):
        row = {"dataset": dataset, "kind": kind, "value": val,
               "variant": v, "tag": tag, "n_seeds": len(g)}
        for m in ["PR-AUC(macro)", "F1-score", "MCC", "Accuracy"]:
            row[f"{m}_mean"] = round(g[m].mean(), 4)
            row[f"{m}_std"] = round(g[m].std(), 4)
        rows.append(row)

    # --- 차수 스윕이면 분해 품질도 기록
    if kind == "degree" and "m4" in a.variants:
        sh(["-m", "src.degree_analysis", *ds, "--variant", "m4",
            "--tag", tag, "--max-degree", str(val)], f"degree {tag}")
        pur = C.TABLE_DIR / f"purity_{dataset}_{tag}_m4.csv"
        # K=1 이면 검정할 고차항이 없어 빈 파일이 되므로 건너뛴다.
        if pur.exists() and pur.stat().st_size > 0:
            try:
                pd_ = pd.read_csv(pur)
            except pd.errors.EmptyDataError:
                pd_ = None
            if pd_ is not None and len(pd_):
                for r in rows:
                    if r["variant"] == "m4":
                        r["max_purity_err"] = float(pd_["relative_error"].max())
                        r["all_homogeneous"] = bool(pd_["is_homogeneous"].all())
        con = C.TABLE_DIR / f"degree_contrib_{dataset}_{tag}_m4.csv"
        if con.exists():
            cd = pd.read_csv(con)
            allrow = cd[cd.Attack == "ALL"]
            if len(allrow):
                for k in range(1, int(val) + 1):
                    col = f"share_deg{k}"
                    if col in allrow:
                        for r in rows:
                            if r["variant"] == "m4":
                                r[col] = round(float(allrow[col].iloc[0]), 4)

    # --- 스케일러 스윕이면 배포 안정성 지표
    if kind in ("scaler", "split", "amplify"):
        mp = C.PROCESSED_DIR / f"{dataset}_{tag}_meta.json"
        if mp.exists():
            m = json.loads(mp.read_text("utf-8"))
            for r in rows:
                r["oob_value_pct"] = m.get("oob_value_pct")
                r["oob_row_pct"] = m.get("oob_row_pct")
                r["dyn_range_u4_u1"] = m.get("dynamic_range_u4_u1")
                r["n_features"] = m.get("n_features")

    el = int(time.time() - t0)
    print(f"  [완료] {tag}  ({el // 60}분 {el % 60}초)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--dataset", choices=list(C.DATASETS) + ["all"],
                    default="all")
    ap.add_argument("--variants", nargs="+", default=["m4"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--full", action="store_true", default=True)
    ap.add_argument("--sampled", dest="full", action="store_false")
    ap.add_argument("--base-tag", default="full",
                    help="기준 전처리 tag (차수 스윕이 재사용)")
    ap.add_argument("--scaler", default=C.SCALER, help="고정할 스케일러")
    ap.add_argument("--split", default="temporal", help="고정할 분할")
    a = ap.parse_args()

    names = list(C.DATASETS) if a.dataset == "all" else [a.dataset]
    t0 = time.time()
    print(f"\n{'#' * 70}")
    print(f"  스윕: {a.kind} = {a.values}")
    print(f"  데이터셋 {names}   변형 {a.variants}   시드 {a.seeds}")
    print(f"{'#' * 70}")

    rows = []
    for ds in names:
        for v in a.values:
            print(f"\n--- {ds} / {a.kind}={v} ---", flush=True)
            r = run_one(ds, a.kind, v, a, a.base_tag)
            if r:
                rows.extend(r)

    if not rows:
        sys.exit("\n[중단] 수집된 결과가 없습니다.")

    df = pd.DataFrame(rows)
    out = C.TABLE_DIR / f"sweep_{a.kind}.csv"
    if out.exists():
        # 값 하나만 추가 실행해도 기존 행이 사라지지 않도록 병합한다.
        prev = pd.read_csv(out)
        # tag 를 키에 포함해야 같은 K 를 스케일러별로 돌린 결과가 공존한다.
        key = ["dataset", "value", "variant", "tag"]
        if all(k in prev.columns for k in key):
            prev["value"] = prev["value"].astype(str)
            df["value"] = df["value"].astype(str)
            merged = pd.concat([prev, df], ignore_index=True)
            df = merged.drop_duplicates(subset=key, keep="last")
            df = df.sort_values(key).reset_index(drop=True)
    df.to_csv(out, index=False)

    print(f"\n{'#' * 70}\n  스윕 결과: {a.kind}\n{'#' * 70}")
    base = ["dataset", "value", "variant", "PR-AUC(macro)_mean",
            "PR-AUC(macro)_std", "F1-score_mean", "MCC_mean"]
    extra = [c for c in ["max_purity_err", "share_deg1", "share_deg2",
                         "share_deg3", "share_deg4", "share_deg5",
                         "share_deg6", "oob_row_pct", "dyn_range_u4_u1"]
             if c in df.columns]
    print(df[base + extra].to_string(index=False))

    el = int(time.time() - t0)
    print(f"\n  경과 {el // 3600}시간 {(el % 3600) // 60}분")
    print(f"  [save] {out.name}")
    print(f"  엑셀에 반영하려면: python -m src.report\n")


if __name__ == "__main__":
    main()
