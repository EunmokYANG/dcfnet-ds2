#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_figures_v2.py  --  Figures 3, 4 and 6 of the paper (DCFNet-DS)

Location : <project_root>/src/p2_figures_v2.py
Run      : python -m src.p2_figures_v2 --inspect   (list rows/columns first)
           python -m src.p2_figures_v2
           python -m src.p2_figures_v2 --base-tag full

Inputs (all written by src/sweep.py and src/degree_analysis.py):
  outputs/tables/sweep_degree.csv
      dataset, kind, value, variant, tag, n_seeds,
      PR-AUC(macro)_mean, PR-AUC(macro)_std, ...,
      max_purity_err, all_homogeneous, share_deg1 ... share_deg6
  outputs/tables/degree_contrib_{dataset}_{tag}_{variant}.csv
      Attack (one row per class plus "ALL"), share_deg1 ... share_degK
  outputs/tables/purity_{dataset}_{tag}_m4.csv
      relative_error, is_homogeneous          (used by --inspect only)

Outputs:
  outputs/paper2_degree/figures/p2_fig1_k_sweep.pdf|.png
  outputs/paper2_degree/figures/p2_fig2_tradeoff.pdf|.png
  outputs/paper2_degree/figures/p2_fig4_class_degree_heatmap.pdf|.png
  one *.source.json manifest beside each figure

IMPORTANT
  sweep_degree.csv holds one row per (dataset, value, variant, tag) and
  src/sweep.py merges on that key with keep="last". A sweep run under a tag
  already present replaces the committed rows for that tag. Keep BASE_TAG
  below as it is unless you mean to overwrite them.
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG -- edit here only
# ============================================================================
CFG = {
    # --- input paths -------------------------------------------------------
    "TABLE_DIR": "outputs/tables",
    "SWEEP_CSV": "outputs/tables/sweep_degree.csv",

    # Preprocessing tag the degree sweep was run against.
    # sweep.py builds tags as "{BASE_TAG}_degree{K}"; the saved models are
    # named {dataset}_full_degree{K}_m4.keras, so the base tag is "full".
    "BASE_TAG": "full",

    # --- output ------------------------------------------------------------
    "OUT_DIR":   "outputs/paper2_degree/figures",
    "PAPER_TAG": "p2",              # filename prefix

    # --- sweep_degree.csv columns -----------------------------------------
    "COL_DATASET":    "dataset",
    "COL_KIND":       "kind",
    "COL_K":          "value",      # sweep axis value = maximum degree
    "COL_VARIANT":    "variant",
    "COL_TAG":        "tag",
    "COL_NSEEDS":     "n_seeds",
    "COL_SCORE_MEAN": "PR-AUC(macro)_mean",
    "COL_SCORE_STD":  "PR-AUC(macro)_std",
    "COL_SHARE_FMT":  "share_deg%d",

    # --- degree_contrib_*.csv ---------------------------------------------
    "COL_ATTACK":  "Attack",
    "ROW_ALL":     "ALL",
    # First column found here is read as the non-decomposable share of m4m.
    "SHARE_OTHER_CANDIDATES": ["share_mlp", "share_other", "share_nonpoly"],

    # --- variants ----------------------------------------------------------
    "BASE_VARIANT": "m4",           # decomposable model
    "MLP_VARIANT":  "m4m",          # model with the non-decomposable branch

    # --- figure settings ---------------------------------------------------
    "K_FOR_HEATMAP":  6,            # Figure 4
    "K_FOR_TRADEOFF": 4,            # Figure 2; must match the text of Sec. V-E
    # The m4/m4g runs of Table III were trained under the plain "full" tag,
    # not under a degree tag, so Figure 2 reads their degree_contrib files
    # from there. Set to "" to fall back to the degree tag.
    "VARIANT_TAG":    "full",
    "DATASET_LABEL": {
        "ciciot2023": "CICIoT2023",
        "nfunsw":     "NF-UNSW-NB15-v3",
    },
}
# ============================================================================

plt.rcParams.update({
    # "font.family" matches src/figures.py so both papers share one typeface.
    "font.family": "serif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
DEG_COLORS = ["#4C72B0", "#8FA8D0", "#C5D0E6", "#F2C0A8", "#DD8452", "#C44E52"]


def dlabel(name):
    return CFG["DATASET_LABEL"].get(str(name), str(name))


def tag_for(k):
    """Reproduce the tag that src/sweep.py builds: {BASE_TAG}_degree{K}."""
    return "%s_degree%d" % (CFG["BASE_TAG"], int(k))


def load_sweep():
    """Read sweep_degree.csv and keep only the paper-2 degree-sweep rows."""
    path = CFG["SWEEP_CSV"]
    if not os.path.exists(path):
        sys.exit("sweep file not found: %s" % path)
    df = pd.read_csv(path)

    if CFG["COL_KIND"] in df.columns:
        df = df[df[CFG["COL_KIND"]].astype(str) == "degree"]
    prefix = CFG["BASE_TAG"] + "_degree"
    df = df[df[CFG["COL_TAG"]].astype(str).str.startswith(prefix)]
    if df.empty:
        sys.exit("No rows with tag prefix '%s'. Run --inspect to see which "
                 "tags exist, then set BASE_TAG in CFG." % prefix)

    df = df.copy()
    df[CFG["COL_K"]] = df[CFG["COL_K"]].astype(int)
    return df


def collect_sweep(df):
    """Return per (dataset, K) mean/std/n for the decomposable variant."""
    sub = df[df[CFG["COL_VARIANT"]] == CFG["BASE_VARIANT"]]
    if sub.empty:
        sys.exit("variant '%s' not present in the sweep rows."
                 % CFG["BASE_VARIANT"])
    cols = [CFG["COL_DATASET"], CFG["COL_K"],
            CFG["COL_SCORE_MEAN"], CFG["COL_SCORE_STD"]]
    out = sub[cols].copy()
    out.columns = ["dataset", "K", "mean", "std"]
    out["n"] = sub[CFG["COL_NSEEDS"]].to_numpy() \
        if CFG["COL_NSEEDS"] in sub.columns else np.nan
    out = out.sort_values(["dataset", "K"]).reset_index(drop=True)

    print("  [sweep] file  : %s" % CFG["SWEEP_CSV"])
    print("  [sweep] tag   : %s_degree*" % CFG["BASE_TAG"])
    print("  [sweep] K     : %s" % sorted(set(out["K"])))
    print("  [sweep] seeds : %s" % sorted(set(out["n"].dropna())))
    print("  [sweep] Table II must quote exactly these values and this n.")
    return out


def collect_shares(df):
    """Return {(dataset, K): array of degree shares} from share_deg* columns."""
    sub = df[df[CFG["COL_VARIANT"]] == CFG["BASE_VARIANT"]]
    shares = {}
    for _, r in sub.iterrows():
        k = int(r[CFG["COL_K"]])
        vals = []
        for deg in range(1, k + 1):
            col = CFG["COL_SHARE_FMT"] % deg
            if col not in sub.columns or pd.isna(r[col]):
                vals = []
                break
            vals.append(float(r[col]))
        if vals:
            shares[(r[CFG["COL_DATASET"]], k)] = np.asarray(vals)
    if not shares:
        print("  [warn] no share_deg* values found; Figure 1 lower panel "
              "will be empty. Re-run the sweep so that degree_analysis "
              "writes degree_contrib_*.csv.")
    return shares


def load_class_shares(dataset, k, variant=None):
    """Read degree_contrib_{dataset}_{tag}_{variant}.csv, excluding the ALL row.

    Returns (class_names, matrix of shape (n_classes, k)) or (None, None).
    """
    variant = variant or CFG["BASE_VARIANT"]
    path = os.path.join(CFG["TABLE_DIR"], "degree_contrib_%s_%s_%s.csv"
                        % (dataset, tag_for(k), variant))
    if not os.path.exists(path):
        return None, None
    d = pd.read_csv(path)
    cols = [CFG["COL_SHARE_FMT"] % deg for deg in range(1, k + 1)]
    cols = [c for c in cols if c in d.columns]
    if not cols:
        return None, None
    d = d[d[CFG["COL_ATTACK"]].astype(str) != CFG["ROW_ALL"]]
    return list(d[CFG["COL_ATTACK"]].astype(str)), d[cols].to_numpy(float)


def load_other_share(dataset, k):
    """Read the non-decomposable share of the MLP variant from the ALL row.

    The m4m checkpoint of Table III was trained under CFG['VARIANT_TAG'],
    which is not the degree-sweep tag; falling back to the degree tag keeps
    the function usable if that ever changes.
    """
    tags = [t for t in (CFG.get("VARIANT_TAG"), tag_for(k)) if t]
    path = None
    for t in tags:
        cand = os.path.join(CFG["TABLE_DIR"], "degree_contrib_%s_%s_%s.csv"
                            % (dataset, t, CFG["MLP_VARIANT"]))
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        print("  [warn] no degree_contrib file for %s/%s; looked for tags %s"
              % (dataset, CFG["MLP_VARIANT"], tags))
        return None

    d = pd.read_csv(path)
    row = d[d[CFG["COL_ATTACK"]].astype(str) == CFG["ROW_ALL"]]
    if row.empty:
        print("  [warn] %s has no '%s' row" % (path, CFG["ROW_ALL"]))
        return None
    for c in CFG["SHARE_OTHER_CANDIDATES"]:
        if c in d.columns and not pd.isna(row[c].iloc[0]):
            return float(row[c].iloc[0])
    cols = [CFG["COL_SHARE_FMT"] % deg for deg in range(1, k + 1)]
    cols = [c for c in cols if c in d.columns]
    if cols:
        return float(max(0.0, 1.0 - row[cols].to_numpy(float).sum()))
    print("  [warn] %s has no share column" % path)
    return None


def inspect():
    print("=" * 74)
    print("SWEEP_CSV :", CFG["SWEEP_CSV"])
    if os.path.exists(CFG["SWEEP_CSV"]):
        df = pd.read_csv(CFG["SWEEP_CSV"])
        print("  shape   :", df.shape)
        print("  columns :", list(df.columns))
        if CFG["COL_TAG"] in df.columns:
            print("  tags    :")
            g = df.groupby([CFG["COL_TAG"], CFG["COL_VARIANT"]]).size()
            print(g.to_string())
        print("\n  --> set CFG['BASE_TAG'] to the prefix above.")
    else:
        print("  !! missing")
    print("-" * 74)
    for pat in ("degree_contrib_*.csv", "purity_*.csv"):
        found = sorted(glob.glob(os.path.join(CFG["TABLE_DIR"], pat)))
        print("%-24s %d file(s)" % (pat, len(found)))
        for p in found[:6]:
            try:
                cols = list(pd.read_csv(p, nrows=1).columns)
            except Exception as e:                 # noqa: BLE001
                cols = ["<unreadable: %s>" % e]
            print("   %-56s %s" % (os.path.basename(p), cols))
    print("=" * 74)


# --------------------------------------------------------------------------
# Figure 1 : performance and degree shares against K
# --------------------------------------------------------------------------
def fig1_k_sweep(sweep, shares):
    datasets = list(dict.fromkeys(sweep["dataset"]))
    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(3.5 * len(datasets), 4.8), sharex=True)
    axes = np.array(axes).reshape(2, len(datasets))

    for j, ds in enumerate(datasets):
        sub = sweep[sweep["dataset"] == ds].sort_values("K")
        ax = axes[0, j]
        ax.errorbar(sub["K"], sub["mean"], yerr=sub["std"].fillna(0),
                    marker="o", ms=4.5, lw=1.4, capsize=3, color="#4C72B0")
        best = sub.loc[sub["mean"].idxmax()]
        ax.scatter([best["K"]], [best["mean"]], s=70, facecolors="none",
                   edgecolors="#C44E52", zorder=5, lw=1.3)
        ax.set_title(dlabel(ds))
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("PR-AUC (macro)")

        ax2 = axes[1, j]
        ks = sorted(k for (d, k) in shares if d == ds)
        if ks:
            bottom = np.zeros(len(ks))
            maxdeg = max(len(shares[(ds, k)]) for k in ks)
            for deg in range(maxdeg):
                vals = np.array([shares[(ds, k)][deg]
                                 if deg < len(shares[(ds, k)]) else 0.0
                                 for k in ks])
                ax2.bar(ks, vals, bottom=bottom, width=0.62,
                        color=DEG_COLORS[deg % len(DEG_COLORS)],
                        edgecolor="white", lw=0.4,
                        label="degree %d" % (deg + 1))
                bottom += vals
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, "no share_deg* data", ha="center", va="center",
                     transform=ax2.transAxes, color="#888")
        ax2.set_xlabel("Maximum interaction order $K$")
        ax2.grid(axis="x", visible=False)
        if j == 0:
            ax2.set_ylabel("Share of logit")
        if j == len(datasets) - 1 and ks:
            ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                       frameon=False)
    save(fig, "fig1_k_sweep")


# --------------------------------------------------------------------------
# Figure 2 : accuracy against decomposability
# --------------------------------------------------------------------------
def fig2_tradeoff(df):
    """Accuracy against decomposability, using the Table III runs.

    Scores come from seeds_{dataset}_{VARIANT_TAG}.csv rather than from the
    degree sweep: m4m and m4g were never part of that sweep.
    """
    k = CFG["K_FOR_TRADEOFF"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    marks = {}
    drawn = 0

    for i, ds in enumerate(sorted(set(df[CFG["COL_DATASET"]]))):
        marks[ds] = ["o", "s", "^", "D"][i % 4]
        spath = os.path.join(CFG["TABLE_DIR"], "seeds_%s_%s.csv"
                             % (ds, CFG["VARIANT_TAG"]))
        if not os.path.exists(spath):
            print("  [warn] %s not found; skipping %s" % (spath, ds))
            continue
        sd = pd.read_csv(spath)

        pts = {}
        for var in (CFG["BASE_VARIANT"], CFG["MLP_VARIANT"]):
            r = sd[sd["Variant"] == var]
            if r.empty:
                print("  [warn] variant %s absent from %s" % (var, spath))
                continue
            y = float(r["PR-AUC(macro)"].mean())
            x = 0.0 if var == CFG["BASE_VARIANT"] else load_other_share(ds, k)
            if x is None:
                continue
            pts[var] = (x, y)
            ax.scatter(x, y, s=52, marker=marks[ds], lw=1.4,
                       edgecolors="#4C72B0",
                       facecolors="none" if var == CFG["MLP_VARIANT"]
                       else "#4C72B0")
            ax.annotate("%s / %s" % (dlabel(ds), var), (x, y), fontsize=7.5,
                        textcoords="offset points", xytext=(7, -3))
            drawn += 1
        if len(pts) == 2:
            ax.annotate("", xy=pts[CFG["MLP_VARIANT"]],
                        xytext=pts[CFG["BASE_VARIANT"]],
                        arrowprops=dict(arrowstyle="->", lw=1.2,
                                        color="#C44E52"))

    if not drawn:
        plt.close(fig)
        print("  [fig2] nothing to plot -- skipped")
        return
    ax.set_xlabel("Share of logit absorbed by the non-decomposable component")
    ax.set_ylabel("PR-AUC (macro)")
    ax.set_xlim(-0.03, 0.72)
    save(fig, "fig2_tradeoff")


# --------------------------------------------------------------------------
# Figure 4 : class x degree contribution heat map
# --------------------------------------------------------------------------
def fig4_heatmap(datasets):
    k = CFG["K_FOR_HEATMAP"]
    data = {}
    for ds in datasets:
        classes, M = load_class_shares(ds, k)
        if classes:
            data[ds] = (classes, M)
    if not data:
        print("  [fig4] no degree_contrib_*_%s_m4.csv -- skipped" % tag_for(k))
        return

    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (ds, (classes, M)) in zip(axes, sorted(data.items())):
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=0,
                       vmax=float(np.nanmax(M)))
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(["deg %d" % (i + 1) for i in range(M.shape[1])],
                           rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_title(dlabel(ds))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig4_class_degree_heatmap")


# --------------------------------------------------------------------------
def write_manifest(path):
    """Record which sweep rows produced the figure. Needed to reproduce."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                              # noqa: BLE001
        commit = "unknown"
    meta = {
        "paper": CFG["PAPER_TAG"],
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "sweep_csv": CFG["SWEEP_CSV"],
        "base_tag": CFG["BASE_TAG"],
        "tag_pattern": "%s_degree{K}" % CFG["BASE_TAG"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save(fig, name):
    os.makedirs(CFG["OUT_DIR"], exist_ok=True)
    stem = "%s_%s" % (CFG["PAPER_TAG"], name)
    for ext in ("pdf", "png"):
        p = os.path.join(CFG["OUT_DIR"], "%s.%s" % (stem, ext))
        fig.savefig(p)
        print("  [save]", os.path.basename(p))
    plt.close(fig)
    write_manifest(os.path.join(CFG["OUT_DIR"], "%s.source.json" % stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="list available tags and columns, then exit")
    ap.add_argument("--only", choices=["1", "2", "4"], default=None)
    ap.add_argument("--base-tag", default=None,
                    help="override CFG['BASE_TAG'] (paper-2 preprocessing tag)")
    ap.add_argument("--sweep-csv", default=None,
                    help="override CFG['SWEEP_CSV']")
    args = ap.parse_args()

    if args.base_tag:
        CFG["BASE_TAG"] = args.base_tag
    if args.sweep_csv:
        CFG["SWEEP_CSV"] = args.sweep_csv

    if args.inspect:
        inspect()
        return

    df = load_sweep()
    datasets = sorted(set(df[CFG["COL_DATASET"]]))
    if args.only in (None, "1"):
        fig1_k_sweep(collect_sweep(df), collect_shares(df))
    if args.only in (None, "2"):
        fig2_tradeoff(df)
    if args.only in (None, "4"):
        fig4_heatmap(datasets)
    print("  output: %s" % CFG["OUT_DIR"])


if __name__ == "__main__":
    main()