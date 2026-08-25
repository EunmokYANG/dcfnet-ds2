#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_paired_stats_v1.py  --  paired significance tests for Table VI of the paper.

Every variant in seeds_{stem}.csv was trained on the same data, the same
split and the same five seeds, so the comparison is paired. A paired test on
the seed-wise differences has more power than the unpaired Welch test the
draft currently reports, and Cohen's d_z is the matching effect size. This
script produces both, plus an exact paired permutation test, which does not
assume normality and is the safer choice at n = 5.

Location : <project_root>/src/p2_paired_stats_v1.py
Run      : python -m src.p2_paired_stats_v1 --dataset nfunsw --tag full
           python -m src.p2_paired_stats_v1 --dataset ciciot2023 --tag full \\
                  --metric F1-score
Output   : outputs/paper2_degree/tables/p2_paired_{stem}.csv

Read-only apart from that file.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

OUT_DIR = "outputs/paper2_degree/tables"


def paired_permutation_p(diff):
    """Exact two-sided permutation test over the 2**n sign flips of diff.

    With five seeds there are only 32 sign assignments, so the p-value is
    computed exactly rather than sampled. The smallest attainable two-sided
    p-value is 2/2**n = 0.0625 at n = 5, which is itself worth reporting:
    no paired comparison at this sample size can reach 0.05 by permutation
    alone, whatever the effect size.
    """
    n = len(diff)
    obs = abs(diff.mean())
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((diff * np.asarray(signs)).mean()) >= obs - 1e-15:
            count += 1
    return count / 2 ** n, 2.0 / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--metric", default="PR-AUC(macro)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}" if a.tag else a.dataset
    path = C.TABLE_DIR / f"seeds_{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. 먼저 src.multiseed 를 실행하세요.")
    df = pd.read_csv(path)
    m = a.metric

    wide = df.pivot(index="seed", columns="Variant", values=m)
    wide = wide.dropna(axis=1, how="any")
    variants = list(wide.columns)
    n = len(wide)
    if n < 2 or len(variants) < 2:
        raise SystemExit("need at least two variants sharing at least two seeds")

    print(f"\n{'=' * 78}\n  {stem}   지표: {m}   공유 시드 {n}개: "
          f"{list(wide.index)}\n{'=' * 78}")
    print(wide.round(4).to_string())

    order = wide.mean().sort_values(ascending=False).index.tolist()
    rows = []
    for v1, v2 in itertools.combinations(order, 2):
        d = (wide[v1] - wide[v2]).to_numpy(float)
        t, p_t = stats.ttest_rel(wide[v1], wide[v2])
        sd = d.std(ddof=1)
        dz = d.mean() / sd if sd > 0 else np.inf
        p_perm, p_floor = paired_permutation_p(d)
        rows.append({"A": v1, "B": v2, "n_pairs": n,
                     "mean_diff": d.mean(), "sd_diff": sd,
                     "t_paired": t, "p_paired_t": p_t,
                     "p_permutation": p_perm,
                     "p_permutation_floor": p_floor,
                     "cohens_dz": dz})

    out = pd.DataFrame(rows)
    k = len(out)
    out["family_size"] = k
    out["p_paired_t_bonferroni"] = (out["p_paired_t"] * k).clip(upper=1.0)
    out["significant_bonferroni"] = out["p_paired_t_bonferroni"] < a.alpha

    print(f"\n{'=' * 78}\n  대응표본 검정 (쌍 {k}개, Bonferroni 임계 "
          f"p < {a.alpha / k:.5f})\n{'=' * 78}")
    print(f"  {'비교':<22}{'평균차':>10}{'t':>9}{'p_t':>11}"
          f"{'p_perm':>10}{'d_z':>9}")
    print("  " + "-" * 74)
    for _, r in out.iterrows():
        print(f"  {r['A'] + ' vs ' + r['B']:<22}{r['mean_diff']:>10.4f}"
              f"{r['t_paired']:>9.2f}{r['p_paired_t']:>11.6f}"
              f"{r['p_permutation']:>10.4f}{r['cohens_dz']:>9.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    dest = os.path.join(OUT_DIR, f"p2_paired_{stem}.csv")
    out.to_csv(dest, index=False)

    print(f"\n  참고: 시드 {n}개에서 순열검정이 도달할 수 있는 최소 양측 p 는 "
          f"{out['p_permutation_floor'].iloc[0]:.4f} 이다.")
    print("  이 값이 보정 임계보다 크면, 어떤 효과 크기에서도 순열검정만으로는")
    print("  유의 판정이 나올 수 없다. 그 경우 t-검정 결과와 d_z 를 함께 보고하고")
    print("  시드 수를 늘리는 편이 낫다.")
    print(f"\n  [save] {dest}\n")


if __name__ == "__main__":
    main()