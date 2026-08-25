#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_arity_v1.py  --  Table IX and Figure 5 of the paper: interaction arity
within each polynomial degree.

A degree-k monomial is a multiset over the features, so it may repeat one:
x_i^2 is second degree over a single feature. Only a monomial whose k indices
are distinct is a genuine k-way conjunction. This script classifies the
coefficients extracted by p2_coeff_attribution_v1.py by that count.

Location : <project_root>/src/p2_arity_v1.py
Run      : python -m src.p2_arity_v1
           python -m src.p2_arity_v1 --topn 8

Input    : outputs/paper2_degree/tables/p2_table6_coeff_attribution.csv
           outputs/paper2_degree/tables/p2_table6_coeff_attribution_nfunsw.csv
Output   : outputs/paper2_degree/tables/p2_table9_arity.csv
           outputs/paper2_degree/figures/p2_fig5_arity.pdf and .png

Read-only apart from those three files.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TABLE_DIR = "outputs/paper2_degree/tables"
FIG_DIR = "outputs/paper2_degree/figures"

DATASETS = [("p2_table6_coeff_attribution.csv", "CICIoT2023"),
            ("p2_table6_coeff_attribution_nfunsw.csv", "NF-UNSW-NB15-v3")]

COL = ["#C44E52", "#DD8452", "#A6C0E8", "#4C72B0"]     # arity 1..4

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})


def load(path):
    """Read one coefficient table and add the arity of each monomial."""
    if not os.path.exists(path):
        raise SystemExit("%s not found. Run src.p2_coeff_attribution_v1 first."
                         % path)
    d = pd.read_csv(path)
    d["arity"] = d["indices"].map(lambda s: len(set(str(s).split("-"))))
    return d


def table9(frames):
    """Full-arity share per degree, by count and weighted by |coefficient|."""
    rows = []
    for name, d in frames:
        for k in sorted(d["degree"].unique()):
            g = d[d["degree"] == k]
            full = g["arity"] == k
            tot = g["abs_coefficient"].sum()
            rows.append({
                "dataset": name,
                "degree": int(k),
                "n_monomials": int(len(g)),
                "full_arity_share_by_count": round(float(full.mean()), 4),
                "full_arity_share_by_weight":
                    round(float(g.loc[full, "abs_coefficient"].sum() / tot), 4)
                    if tot else np.nan,
            })
    return pd.DataFrame(rows)


def figure5(frames, out_stem):
    fig, axes = plt.subplots(1, len(frames), figsize=(7.0, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, frames):
        degs = sorted(d["degree"].unique())
        bottom = np.zeros(len(degs))
        for a in range(1, max(degs) + 1):
            vals = np.array([(d[d.degree == k]["arity"] == a).mean()
                             for k in degs])
            ax.bar(degs, vals, bottom=bottom, width=0.62,
                   color=COL[(a - 1) % len(COL)], edgecolor="white",
                   linewidth=0.6, label="arity %d" % a)
            bottom += vals
        wfull = []
        for k in degs:
            g = d[d.degree == k]
            tot = g["abs_coefficient"].sum()
            wfull.append(g.loc[g.arity == k, "abs_coefficient"].sum() / tot
                         if tot else np.nan)
        ax.plot(degs, wfull, marker="o", ms=4.5, lw=1.4, color="#111111",
                zorder=5)
        for k, v in zip(degs, wfull):
            ax.annotate("%.0f%%" % (100 * v), (k, v),
                        textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7.5, color="#111111")
        ax.set_xticks(degs)
        ax.set_xlabel("Polynomial degree $k$")
        ax.set_ylim(0, 1.02)
        ax.set_title(name)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Share of extracted monomials")
    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ms=4.5, lw=1.4, color="#111111"))
    l.append("full arity,\nweighted by\n|coefficient|")
    axes[-1].legend(h, l, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, title="distinct features")
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = "%s.%s" % (out_stem, ext)
        fig.savefig(path)
        print("  [save]", path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=None,
                    help="keep only the N largest coefficients per class and "
                         "degree; the extraction already applies its own --topn")
    a = ap.parse_args()

    frames = []
    for fname, label in DATASETS:
        d = load(os.path.join(TABLE_DIR, fname))
        if a.topn:
            d = (d.sort_values("abs_coefficient", ascending=False)
                   .groupby(["class", "degree"]).head(a.topn))
        frames.append((label, d))
        print("%-18s %d rows, degrees %s"
              % (label, len(d), sorted(d["degree"].unique())))

    t = table9(frames)
    os.makedirs(TABLE_DIR, exist_ok=True)
    out = os.path.join(TABLE_DIR, "p2_table9_arity.csv")
    t.to_csv(out, index=False)
    print("\n[Table IX]")
    print(t.to_string(index=False))
    print("  [save]", out)

    figure5(frames, os.path.join(FIG_DIR, "p2_fig5_arity"))
    print("\nThe shares describe the extracted coefficients only, not the "
          "whole polynomial; Section V-F states that limit.")


if __name__ == "__main__":
    main()