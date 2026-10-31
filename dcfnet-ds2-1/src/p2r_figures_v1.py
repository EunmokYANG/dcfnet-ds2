"""p2r_figures_v1.py -- figures 3 to 7 of the revised manuscript (v19).

Reads the CSV files written by src.p2r_analysis_v2 (outputs/p2r/analysis)
and writes vector PDFs with embedded TrueType fonts (no Type 3 fonts):

  fig3_degree_sweep.pdf      macro PR-AUC vs K, ten seeds, with the
                             parameter-matched degree-2 models and mlp_s
  fig5_shares.pdf            pooled degree shares at K = 4: logit magnitude
                             (Eq. 10) vs true-class margin, mean +- sd
  fig6_arity.pdf             distinct-feature share of the largest monomials
                             vs combinatorial and population references
  fig4_class_gain.pdf        per-class PR-AUC change from K = 2 to K = 6,
                             paired over the ten seeds
  fig7_frontier.pdf          macro PR-AUC vs model size for every model

Run:
  python -m src.p2r_figures_v1 --analysis outputs/p2r/analysis --out outputs/p2r/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
})
DS = [("ciciot2023_p2r", "CICIoT2023"), ("nfunsw_p2r", "NF-UNSW-NB15-v3")]
C1, C2, C3, C4 = "#1F4E8C", "#C8553D", "#5B8C5A", "#8A93A3"


def fig3(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharex=True)
    for ax, (stem, name) in zip(axes, DS):
        s = pd.read_csv(a / f"summary_{stem}.csv").set_index("config")
        ks = [2, 3, 4, 5, 6]
        m = [s.loc[f"m4_K{k}", "pr_auc_macro_mean"] for k in ks]
        e = [s.loc[f"m4_K{k}", "pr_auc_macro_sd"] for k in ks]
        ax.errorbar(ks, m, yerr=e, color=C1, marker="o", ms=3.2, capsize=2,
                    label="DCFNet-DS, degree $\\leq K$")
        for k, cfg in [(4, "m2w_H3"), (6, "m2w_H5")]:
            ax.errorbar([k + 0.12], [s.loc[cfg, "pr_auc_macro_mean"]],
                        yerr=[s.loc[cfg, "pr_auc_macro_sd"]], color=C2,
                        marker="s", ms=3.2, mfc="white", capsize=2, ls="none",
                        label="degree 2, matched parameters"
                        if k == 4 else None)
        ax.errorbar([4 - 0.12], [s.loc["mlp_s", "pr_auc_macro_mean"]],
                    yerr=[s.loc["mlp_s", "pr_auc_macro_sd"]], color=C3,
                    marker="^", ms=3.4, mfc="white", capsize=2, ls="none",
                    label="MLP, matched parameters")
        ax.set_title(name)
        ax.set_xticks(ks)
        ax.set_xlabel("Maximum degree $K$")
        ax.grid(True, alpha=0.5)
    axes[0].set_ylabel("Macro PR-AUC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.6)
    fig.savefig(out / "fig3_degree_sweep.pdf")
    plt.close(fig)


def fig4(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
    for ax, (stem, name) in zip(axes, DS):
        d = pd.read_csv(a / f"shares_summary_{stem}.csv")
        r = d[(d.variant == "m4") & (d.K == 4) & (d["class"] == "ALL")].iloc[0]
        x = np.arange(1, 5)
        w = 0.36
        for off, pfx, col, lab in [(-w / 2, "", C1, "logit magnitude"),
                                   (w / 2, "margin_", C2,
                                    "true-class margin")]:
            ax.bar(x + off, [r[f"{pfx}share_deg{k}_mean"] for k in x], w,
                   yerr=[r[f"{pfx}share_deg{k}_sd"] for k in x], color=col,
                   ecolor="black", error_kw={"elinewidth": 0.5, "capsize": 1.5},
                   label=lab)
        ax.set_xticks(x)
        ax.set_xlabel("Degree $k$ ($K=4$)")
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].set_ylabel("Share")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=0.6)
    fig.savefig(out / "fig5_shares.pdf")
    plt.close(fig)


def fig5(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
    for ax, (stem, name) in zip(axes, DS):
        d = pd.read_csv(a / f"arity_null_{stem}_K4.csv")
        g = d[d.degree >= 2].groupby("degree").mean(numeric_only=True)
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.obs_count_share, 0.36, color=C1,
               label="top-8 monomials (observed)")
        ax.bar(x + 0.2, g.pop_weighted_share, 0.36, color=C4,
               label="all monomials, |coef.|-weighted")
        ax.plot(x, g.null_uniform_multiset, ls="none", marker="D", ms=3.6,
                color=C2, label="combinatorial reference")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(k)) for k in g.index])
        ax.set_xlabel("Degree ($K=4$)")
        ax.set_ylim(0.5, 1.02)
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].set_ylabel("Distinct-feature share")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03), columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.6)
    fig.savefig(out / "fig6_arity.pdf")
    plt.close(fig)


def fig6(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2),
                             gridspec_kw={"width_ratios": [8, 10]},
                             sharey=True)
    for ax, (stem, name) in zip(axes, DS):
        pc = pd.read_csv(a / f"perclass_{stem}.csv")
        pc = pc[pc.variant == "m4"]
        p = pc.pivot_table(index=["class", "seed"], columns="K",
                           values="pr_auc")
        d = (p[6] - p[2]).groupby("class")
        m, s = d.mean(), d.std()
        cls = list(m.index)
        x = np.arange(len(cls))
        ax.bar(x, m.values, 0.62, yerr=s.values,
               color=[C1 if v >= 0 else C2 for v in m.values],
               ecolor="black", error_kw={"elinewidth": 0.5, "capsize": 1.5})
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(cls, rotation=30, ha="right")
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.5)
    axes[0].set_ylabel("$\\Delta$ PR-AUC ($K$=6 $-$ $K$=2)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "fig4_class_gain.pdf")
    plt.close(fig)


NAMES = {"lr": "LR", "lr_l1_tuned": "L1-LR", "poly2_l1_tuned": "poly-2 L1",
         "dt_d4": "DT d4", "dt_d8": "DT d8", "rf_small": "RF small",
         "rf": "RF", "hgb_small": "HGB small", "hgb": "HGB", "ebm": "EBM",
         "m4_K4": "DCFNet $K$=4", "m4_K6": "DCFNet $K$=6",
         "m4m_K4": "m4m", "nn_mlp": "MLP"}
# label offsets in points, chosen so that labels do not overlap
OFFS = {"ciciot2023_p2r": {"hgb": (4, -7), "hgb_small": (-30, 3),
                           "ebm": (-14, 4), "rf": (3, -8),
                           "m4_K4": (-52, -2), "m4_K6": (-6, -10),
                           "m4m_K4": (-4, 4), "nn_mlp": (3, -8)},
        "nfunsw_p2r": {"hgb": (2, 3), "hgb_small": (-36, 2),
                       "ebm": (3, -6), "rf": (-8, 3),
                       "m4_K4": (-22, -9), "m4_K6": (-4, 4),
                       "m4m_K4": (2, -8), "nn_mlp": (-6, 4),
                       "lr": (3, 1), "lr_l1_tuned": (3, -6)}}


def fig7(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for ax, (stem, name) in zip(axes, DS):
        f = pd.read_csv(a / f"frontier_{stem}.csv").set_index("model")
        s = pd.read_csv(a / f"summary_{stem}.csv").set_index("config")
        pts = []
        for mdl in ["lr", "lr_l1_tuned", "poly2_l1_tuned", "dt_d4", "dt_d8",
                    "rf_small", "rf", "hgb_small", "hgb", "ebm"]:
            pts.append((mdl, f.loc[mdl, "params_or_nodes"],
                        f.loc[mdl, "pr_auc_macro_mean"], "other"))
        for cfg in ["m4_K4", "m4_K6"]:
            pts.append((cfg, s.loc[cfg, "params_mean"],
                        s.loc[cfg, "pr_auc_macro_mean"], "exact"))
        for cfg in ["m4m_K4", "nn_mlp"]:
            pts.append((cfg, s.loc[cfg, "params_mean"],
                        s.loc[cfg, "pr_auc_macro_mean"], "neural"))
        for mdl, x, y, kind in pts:
            st = {"exact": dict(marker="o", color=C1, ms=5),
                  "neural": dict(marker="s", color=C3, ms=4.2, mfc="white"),
                  "other": dict(marker="^", color=C4, ms=4.2)}[kind]
            ax.plot([x], [y], ls="none", **st)
            ax.annotate(NAMES[mdl], (x, y),
                        xytext=OFFS[stem].get(mdl, (3, 2)),
                        textcoords="offset points", fontsize=6.3)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (parameters, tree nodes, or stored bins)")
        ax.set_title(name)
        ax.grid(True, alpha=0.5)
    axes[0].set_ylabel("Macro PR-AUC")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker="o", color=C1, ms=5),
          Line2D([], [], ls="none", marker="s", color=C3, ms=4.2, mfc="white"),
          Line2D([], [], ls="none", marker="^", color=C4, ms=4.2)]
    fig.legend(hs, ["exactly decomposable by polynomial degree",
                    "neural, not decomposable by degree",
                    "linear, additive and tree baselines"],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig7_frontier.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="outputs/p2r/analysis")
    ap.add_argument("--out", default="outputs/p2r/figures")
    a = ap.parse_args()
    src, out = Path(a.analysis), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig3, fig4, fig5, fig6, fig7):
        f(src, out)
        print("written", f.__name__, flush=True)


if __name__ == "__main__":
    main()
