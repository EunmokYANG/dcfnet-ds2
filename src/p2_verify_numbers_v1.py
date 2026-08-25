#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_verify_numbers_v1.py  --  seed counts and significance tests for the paper.

Location : <project_root>/src/p2_verify_numbers_v1.py
Run      : python -m src.p2_verify_numbers_v1 seeds --csv outputs/tables/seeds_ciciot2023_full_degree4.csv
           python -m src.p2_verify_numbers_v1 ttest --csv outputs/tables/seeds_ciciot2023_full.csv \
                  --a m4 --b m4m --family 4

Input is the per-seed table written by src/multiseed.py:
    outputs/tables/seeds_{dataset}_{tag}.csv
    columns: Variant, seed, PR-AUC(macro), F1-score, MCC, Accuracy
One file per (dataset, tag), so the dataset is fixed by the file chosen.
Do NOT pass outputs/tables/sweep_degree.csv here: that table is already
aggregated over seeds and carries no per-seed values.

  seeds   how many seeds back each variant. Table II and Table III of the
          paper must not quote different seed counts.
  ttest   Welch t-test and Cohen's d for one pair of variants, with a
          Bonferroni factor for the ablation family.

Everything else this file used to do is now covered elsewhere:
  parameter counts   python -m src.p2_coeff_attribution_v1 --ckpt <model> --summary
  DegreeScale sigma  the same script prints sigma_k in build()
  identity residual  src/degree_analysis.py prints max|sum_k phi_k + b - logit|
  homogeneity        src/degree_analysis.py writes purity_{tag}.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
def cmd_seeds(a):
    """Report how many seeds back each variant in one seeds_*.csv file."""
    df = pd.read_csv(a.csv)
    for col in (a.variant_col, a.score_col):
        if col not in df.columns:
            sys.exit("column '%s' not in %s. Columns are: %s"
                     % (col, a.csv, list(df.columns)))
    g = df.groupby(a.variant_col)[a.score_col].agg(["count", "mean", "std"])
    print("file: %s" % a.csv)
    print(g.round(4).to_string())
    if a.seed_col in df.columns:
        print("\nseeds present: %s"
              % sorted(df[a.seed_col].unique().tolist()))
    print("\nTable II and Table III of the paper must quote the same seed "
          "count, or state the difference in a footnote.")


# --------------------------------------------------------------------------
def cmd_ttest(a):
    """Welch t-test and Cohen's d for two variants in one seeds_*.csv file."""
    from scipy import stats
    df = pd.read_csv(a.csv)
    if a.variant_col not in df.columns:
        sys.exit("column '%s' not in %s. Columns are: %s"
                 % (a.variant_col, a.csv, list(df.columns)))
    xa = df[df[a.variant_col] == a.a][a.score_col].to_numpy(float)
    xb = df[df[a.variant_col] == a.b][a.score_col].to_numpy(float)
    if len(xa) < 2 or len(xb) < 2:
        sys.exit("too few samples: n(%s)=%d, n(%s)=%d"
                 % (a.a, len(xa), a.b, len(xb)))

    t, p = stats.ttest_ind(xb, xa, equal_var=False)
    sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1)
                  + (len(xb) - 1) * xb.var(ddof=1))
                 / (len(xa) + len(xb) - 2))
    d = (xb.mean() - xa.mean()) / sp if sp > 0 else np.inf

    print("file          : %s" % a.csv)
    print("metric        : %s" % a.score_col)
    print("%-12s n=%d  mean=%.4f  sd=%.4f"
          % (a.a, len(xa), xa.mean(), xa.std(ddof=1)))
    print("%-12s n=%d  mean=%.4f  sd=%.4f"
          % (a.b, len(xb), xb.mean(), xb.std(ddof=1)))
    print("delta         : %+.4f" % (xb.mean() - xa.mean()))
    print("Welch t       : %.4f" % t)
    print("p (raw)       : %.6g" % p)
    print("p (Bonferroni x%d): %.6g" % (a.family, min(1.0, p * a.family)))
    print("Cohen d       : %.4f" % d)
    if min(len(xa), len(xb)) < 5:
        print("\n[warn] fewer than five seeds per arm. A non-significant p "
              "here means the test is underpowered, not that the effect is "
              "absent. Report it as undemonstrated, and quote d as well.")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Column names follow src/multiseed.py, which writes seeds_{stem}.csv
    # with the columns Variant, seed, PR-AUC(macro), F1-score, MCC, Accuracy.
    p = sub.add_parser("seeds")
    p.add_argument("--csv", required=True)
    p.add_argument("--variant-col", default="Variant")
    p.add_argument("--seed-col", default="seed")
    p.add_argument("--score-col", default="PR-AUC(macro)")
    p.set_defaults(fn=cmd_seeds)

    p = sub.add_parser("ttest")
    p.add_argument("--csv", required=True)
    p.add_argument("--a", required=True, help="baseline variant, e.g. m4")
    p.add_argument("--b", required=True, help="compared variant, e.g. m4m")
    p.add_argument("--family", type=int, default=4,
                   help="number of comparisons for the Bonferroni factor")
    p.add_argument("--variant-col", default="Variant")
    p.add_argument("--seed-col", default="seed")
    p.add_argument("--score-col", default="PR-AUC(macro)")
    p.set_defaults(fn=cmd_ttest)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
