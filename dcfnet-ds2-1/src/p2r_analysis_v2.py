"""p2r_analysis_v2.py -- post-hoc analyses of the paper-2 resubmission runs.

Reads the checkpoints of src.p2r_train_v2 (outputs/p2r/models) and writes
CSV tables to outputs/p2r/analysis. Every analysis recomputes the model from
its weights in float64 numpy, independently of the TensorFlow graph.

Sub-commands (reviewer comment)
  exact     R1-1, R2-1   identity residual in float32 (TF path), float64
                         layer-free reconstruction, independent monomial-
                         coefficient evaluation (degrees 1-4), reversed and
                         float32 accumulation orders, large-magnitude inputs
  shares    R1-3, R2-3,  Eq.(10) shares with explicit denominator, signed
            R1-5, R2-8   shares, cancellation ratio, true-vs-runner-up margin
                         decomposition, mean-centred logits, degree removal
                         (flip rate and macro PR-AUC), non-decomposable share
  coeff     R2-5, R1-5   top-8 monomials per class and degree with arity,
                         combinatorial / population / permutation nulls,
                         Jaccard stability of the top sets across seeds
  perclass  R1-8, R2-6   per-class precision, recall, PR-AUC, confusion matrix,
                         recall at fixed FPR, bootstrap CIs (block bootstrap
                         over time-ordered NF test rows)
  perturb   R1-9, R1-10  feature-set shuffling with 999 random controls;
                         marginal and dependency-preserving (joint, stratified)
                         replacement; recall drop, flip rate, margin change
  stats     R1-13, R1-12 seed-paired tests between variants (t-test, exact
                         sign-flip permutation, Bonferroni), summary tables,
                         frontier table with retraining stability

Coefficient convention (paper Eq. 13): the readout index is the OUTERMOST
factor and pairs with W_k, i.e. for a path (a_k, ..., a_1)
    c = V_k[c, a_k] W_k[a_k, a_{k-1}] ... W_2[a_2, a_1] / (sigma_1...sigma_k).
src/p2_coeff_attribution_v1.py (v1.0.0) applied W_2 at the outermost hop,
which is correct only for k = 2; `exact` verifies the order used here.

Run:
  python -m src.p2r_analysis_v2 exact    --dataset nfunsw --tag p2r
  python -m src.p2r_analysis_v2 shares   --dataset nfunsw --tag p2r
  python -m src.p2r_analysis_v2 all      --dataset ciciot2023 --tag p2r
"""

import argparse
import itertools
import json
import re
from math import comb

import numpy as np
import pandas as pd

from src import config as C

RUNS = C.OUT_DIR / "p2r" / "runs"
MODELS = C.OUT_DIR / "p2r" / "models"
BASE = C.OUT_DIR / "p2r" / "baselines"
OUT = C.OUT_DIR / "p2r" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
TOPN = 8
N_CONTROLS = 999


# ------------------------------------------------------------------ loading
def load_data(stem):
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads((C.PROCESSED_DIR / f"{stem}_meta.json")
                      .read_text("utf-8"))
    return d, meta


def runs_of(stem, variant=None, K=None):
    """Result JSONs of a dataset tag, optionally filtered."""
    out = []
    for p in sorted(RUNS.glob(f"{stem}_*.json")):
        r = json.loads(p.read_text("utf-8"))
        if r.get("tag") != stem.split("_", 1)[1]:
            continue
        if variant and r["variant"] != variant:
            continue
        if K and r["K"] != K:
            continue
        out.append(r)
    return out


def load_keras(run):
    import keras
    from src.model import CUSTOM_OBJECTS
    return keras.models.load_model(MODELS / f"{run}{C.MODEL_EXT}",
                                   custom_objects=CUSTOM_OBJECTS,
                                   compile=False)


def sigma_of(layer):
    """Debiased DegreeScale divisor used at inference (src/model.py)."""
    s = float(np.asarray(layer.s).reshape(-1)[0])
    step = float(np.asarray(layer.step).reshape(-1)[0])
    debias = 1.0 - layer.momentum ** max(step, 1.0)
    return s / max(debias, layer.eps) + layer.eps


class Weights:
    """float64 copy of a degree-separated model (m4, m4m, m4ms, m2w).

    terms: list of dicts {name, degree, chain}, where chain lists the
    (W_kernel, sigma) hops from degree 1 upward and V is the readout kernel.
    Keras kernels are (in, out): u_k = x * (ubar_{k-1} @ K_k), phi = ubar @ V.
    """

    def __init__(self, model):
        L = {l.name: l for l in model.layers}
        self.b = np.asarray(L["logits"].b, dtype=np.float64)
        self.sig1 = sigma_of(L["scale_deg1"])
        self.terms = [{"name": "deg1", "degree": 1, "chain": [],
                       "V": np.asarray(L["logit_deg1"].kernel, np.float64)}]
        k, chain = 2, []
        while f"logit_deg{k}" in L:                      # m4 family
            chain = chain + [(np.asarray(L[f"cross_deg{k}_w"].kernel,
                                         np.float64),
                              sigma_of(L[f"scale_deg{k}"]))]
            self.terms.append({"name": f"deg{k}", "degree": k,
                               "chain": list(chain),
                               "V": np.asarray(L[f"logit_deg{k}"].kernel,
                                               np.float64)})
            k += 1
        h = 1
        while f"logit_deg2_h{h}" in L:                   # m2w heads
            self.terms.append({
                "name": f"deg2_h{h}", "degree": 2,
                "chain": [(np.asarray(L[f"cross_deg2_h{h}_w"].kernel,
                                      np.float64),
                           sigma_of(L[f"scale_deg2_h{h}"]))],
                "V": np.asarray(L[f"logit_deg2_h{h}"].kernel, np.float64)})
            h += 1
        self.K = max(t["degree"] for t in self.terms)
        self.has_mlp = "logit_mlp" in L
        self.model = model

    def phis(self, x, dtype=np.float64, batch=65_536):
        """Per-term logit contributions, (n_terms, n, C)."""
        out = np.empty((len(self.terms), len(x), len(self.b)), dtype)
        for s in range(0, len(x), batch):
            xb = x[s:s + batch].astype(dtype)
            ub1 = xb / dtype(self.sig1)
            for t, term in enumerate(self.terms):
                u = ub1
                for Wk, sk in term["chain"]:
                    u = xb * (u @ Wk.astype(dtype)) / dtype(sk)
                out[t, s:s + batch] = u @ term["V"].astype(dtype)
        return out

    def mlp_phi(self, x):
        from tensorflow.keras import models
        sub = models.Model(self.model.input,
                           self.model.get_layer("logit_mlp").output)
        return sub.predict(x, batch_size=8192, verbose=0).astype(np.float64)

    def degree_groups(self):
        """Map degree -> indices of terms (m2w heads share degree 2)."""
        g = {}
        for i, t in enumerate(self.terms):
            g.setdefault(t["degree"], []).append(i)
        return g

    def coefficient_tensor(self, term_idx, c):
        """Ordered-path coefficient tensor of one term and class (float64),
        axes ordered (a_k, ..., a_1): readout index first (Eq. 13)."""
        term = self.terms[term_idx]
        coef = term["V"][:, c].copy()                    # over a_k
        sig = self.sig1 * np.prod([sk for _, sk in term["chain"]])
        for Wk, _ in reversed(term["chain"]):            # W_k outermost
            coef = coef[..., None] * Wk.T                # Wpaper = K.T
        return coef / sig


def tf_parts(model, x):
    """TF float32 per-term outputs and logits (the v1 accumulation path)."""
    from tensorflow.keras import models
    names = [l.name for l in model.layers if l.name.startswith("logit_deg")
             or l.name == "logit_mlp"]
    sub = models.Model(model.input, [model.get_layer(n).output
                                     for n in names]
                       + [model.get_layer("logits").output])
    outs = sub.predict(x, batch_size=8192, verbose=0)
    return names, outs[:-1], outs[-1]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_prauc(y, prob, n_cls):
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    return float(average_precision_score(
        label_binarize(y, classes=np.arange(n_cls)), prob, average="macro"))


# ------------------------------------------------------------------ exact
def cmd_exact(stem, a):
    d, meta = load_data(stem)
    x = d["x_test"]
    rows = []
    for r in runs_of(stem):
        if r["variant"] not in ("m4", "m2w"):
            continue
        if r["seed"] > a.max_seed:
            continue
        model = load_keras(r["run"])
        w = Weights(model)
        names, tfp, tflogit = tf_parts(model, x)
        # pair TF outputs with weight terms by layer name, not layer order
        by_name = dict(zip(names, tfp))
        order = [f"logit_{t['name']}" for t in w.terms]
        tfp = np.stack([by_name[n] for n in order])
        # (1) v1 check: same float32 path on both sides
        res_tf = float(np.abs(tfp.sum(0) + w.b.astype(np.float32)
                              - tflogit).max())
        # (2) float64 layer-free reconstruction vs TF logits
        p64 = w.phis(x)
        logit64 = p64.sum(0) + w.b
        dev64 = float(np.abs(logit64 - tflogit).max())
        rel64 = dev64 / float(np.abs(logit64).max())
        # (3) reversed and float32 accumulation orders
        rev32 = (tfp[::-1].astype(np.float32).sum(0)
                 + w.b.astype(np.float32))
        dev_rev = float(np.abs(rev32 - tflogit).max())
        # (4) independent monomial-coefficient evaluation on a subsample
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=min(a.coef_rows, len(x)),
                         replace=False)
        xs = x[idx].astype(np.float64)
        dev_coef = 0.0
        for ti, t in enumerate(w.terms):
            if t["degree"] > 4:
                continue
            for c in range(len(w.b)):
                T = w.coefficient_tensor(ti, c)
                # contract every axis of the coefficient tensor with x
                letters = "abcd"[:t["degree"]]
                spec = letters + "," + ",".join(f"n{l}" for l in letters) \
                    + "->n"
                val = np.einsum(spec, T, *([xs] * t["degree"]),
                                optimize=True)
                dev_coef = max(dev_coef,
                               float(np.abs(val - p64[ti, idx, c]).max()))
        # (5) homogeneity under large inputs, float64 and TF float32
        hom = {}
        for s in (2.0, 10.0, 100.0):
            ps = w.phis(xs * s)
            base = p64[:, idx]
            err = [float(np.abs(ps[ti] - s ** t["degree"] * base[ti]).max()
                         / (np.abs(s ** t["degree"] * base[ti]).max()
                            + 1e-300))
                   for ti, t in enumerate(w.terms)]
            hom[f"hom64_x{s:g}"] = max(err)
            nm_s, tps_s, _ = tf_parts(model, (xs * s).astype(np.float32))
            tps = [dict(zip(nm_s, tps_s))[n] for n in order]
            err32 = [float(np.abs(np.asarray(tps[ti], np.float64)
                                  - s ** t["degree"] * base[ti]).max()
                           / (np.abs(s ** t["degree"] * base[ti]).max()
                              + 1e-300))
                     for ti, t in enumerate(w.terms)]
            hom[f"hom32_x{s:g}"] = max(err32)
        rows.append({"run": r["run"], "variant": r["variant"], "K": r["K"],
                     "seed": r["seed"], "n_flows": len(x),
                     "residual_tf_float32": res_tf,
                     "maxdev_float64_vs_tf": dev64,
                     "reldev_float64_vs_tf": rel64,
                     "maxdev_reversed_order_float32": dev_rev,
                     "maxdev_coeff_tensor_vs_float64": dev_coef,
                     "coeff_rows": len(idx), **hom})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"exact_{stem}.csv", index=False)


# ------------------------------------------------------------------ shares
def share_stats(phi_deg, y, mask):
    """phi_deg: (G, n, C) grouped contributions. Returns dict of shares."""
    sel = np.flatnonzero(mask)
    yt = y[sel]
    pt = phi_deg[:, sel, yt]                              # (G, n) true-class
    absm = np.abs(pt).mean(1)
    out = {"eq10_share": absm / absm.sum(),
           "signed_mean": pt.mean(1),
           "cancellation": float(np.abs(pt.sum(0)).mean()
                                 / np.abs(pt).sum(0).mean())}
    # margin vs strongest competitor
    tot = phi_deg.sum(0)[sel]                             # without bias
    return out, sel, pt, tot


def cmd_shares(stem, a):
    d, meta = load_data(stem)
    x, y = d["x_test"], d["y_test"]
    classes = list(meta["label_mapping"])
    n_cls = meta["n_classes"]
    rows, rem_rows = [], []
    for r in runs_of(stem):
        if r["variant"] not in ("m4", "m2w", "m4m", "m4ms"):
            continue
        model = load_keras(r["run"])
        w = Weights(model)
        p = w.phis(x)
        groups = w.degree_groups()
        degs = sorted(groups)
        G = np.stack([p[groups[k]].sum(0) for k in degs])
        labels = [f"deg{k}" for k in degs]
        if w.has_mlp:
            G = np.concatenate([G, w.mlp_phi(x)[None]], 0)
            labels.append("mlp")
        logit = G.sum(0) + w.b
        prob = softmax(logit)
        pred = prob.argmax(1)
        # runner-up class
        masked = logit.copy()
        masked[np.arange(len(y)), y] = -np.inf
        cstar = masked.argmax(1)
        dphi = (G[:, np.arange(len(y)), y] - G[:, np.arange(len(y)), cstar])
        db = w.b[y] - w.b[cstar]
        margin = dphi.sum(0) + db
        centred = G - G.mean(2, keepdims=True)
        for c, nm in list(enumerate(classes)) + [(None, "ALL")]:
            m = np.ones(len(y), bool) if c is None else (y == c)
            if not m.any():
                continue
            sel = np.flatnonzero(m)
            pt = G[:, sel, y[sel]]
            absm = np.abs(pt).mean(1)
            cen = np.abs(centred[:, sel, y[sel]]).mean(1)
            dm = np.abs(dphi[:, sel]).mean(1)
            row = {"run": r["run"], "variant": r["variant"], "K": r["K"],
                   "seed": r["seed"], "class": nm, "n": int(m.sum()),
                   "denominator": "sum_j mean_x |phi_j,y(x)| over flows of "
                                  "the class, bias excluded",
                   "cancellation_ratio": float(
                       np.abs(pt.sum(0)).mean() / np.abs(pt).sum(0).mean()),
                   "mean_margin": float(margin[sel].mean()),
                   "mean_bias_margin": float(db[sel].mean()),
                   "frac_correct": float((pred[sel] == y[sel]).mean())}
            for j, lab in enumerate(labels):
                row[f"share_{lab}"] = float(absm[j] / absm.sum())
                row[f"signed_{lab}"] = float(pt[j].mean())
                row[f"centred_share_{lab}"] = float(cen[j] / cen.sum())
                row[f"margin_share_{lab}"] = float(dm[j] / dm.sum())
                row[f"margin_signed_{lab}"] = float(dphi[j, sel].mean())
            rows.append(row)
        base = macro_prauc(y, prob, n_cls)
        for j, lab in enumerate(labels):
            pr = softmax(logit - G[j])
            pj = pr.argmax(1)
            rr = {"run": r["run"], "variant": r["variant"], "K": r["K"],
                  "seed": r["seed"], "removed": lab,
                  "flip_rate": float((pj != pred).mean()),
                  "macro_prauc_full": base,
                  "macro_prauc_removed": macro_prauc(y, pr, n_cls)}
            for c, nm in enumerate(classes):
                m = y == c
                if m.any():
                    rr[f"recall_full_{nm}"] = float((pred[m] == c).mean())
                    rr[f"recall_removed_{nm}"] = float((pj[m] == c).mean())
            rem_rows.append(rr)
        print(f"[shares] {r['run']}", flush=True)
    sh = pd.DataFrame(rows)
    sh.to_csv(OUT / f"shares_{stem}.csv", index=False)
    pd.DataFrame(rem_rows).to_csv(OUT / f"removal_{stem}.csv", index=False)
    # seed variability of the shares (R1-5)
    if len(sh):
        num = [c for c in sh.columns if c.startswith(("share_",
                                                      "margin_share_",
                                                      "centred_share_"))]
        g = sh.groupby(["variant", "K", "class"])[num]
        summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
        summ["n_seeds"] = g.size()
        summ.reset_index().to_csv(OUT / f"shares_summary_{stem}.csv",
                                  index=False)


# ------------------------------------------------------------------ coeff
def fold_monomials(T, k):
    """Fold an ordered-path tensor onto multisets. Returns keys, sums, idx."""
    D = T.shape[0]
    idx = np.stack(np.indices((D,) * k, dtype=np.int32), -1).reshape(-1, k)
    idx.sort(1)
    key = np.zeros(len(idx), np.int64)
    for col in range(k):
        key = key * D + idx[:, col]
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=T.reshape(-1), minlength=len(uk))
    first = np.zeros(len(uk), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return uk, sums, idx[first]


def cmd_coeff(stem, a):
    d, meta = load_data(stem)
    feats = [str(f) for f in d["feature_names"]]
    classes = list(meta["label_mapping"])
    D = len(feats)
    rows, null_rows = [], []
    for r in runs_of(stem, "m4", a.coeff_K):
        w = Weights(load_keras(r["run"]))
        for ti, t in enumerate(w.terms):
            k = t["degree"]
            if k > 4:
                continue
            for c, nm in enumerate(classes):
                uk, sums, idx = fold_monomials(w.coefficient_tensor(ti, c), k)
                distinct = np.array([len(set(row)) == k for row in idx])
                order = np.argsort(-np.abs(sums))[:TOPN]
                for rank, j in enumerate(order, 1):
                    rows.append({
                        "run": r["run"], "seed": r["seed"], "class": nm,
                        "degree": k, "rank": rank,
                        "monomial": " x ".join(feats[i] for i in idx[j]),
                        "indices": " ".join(map(str, idx[j])),
                        "coefficient": float(sums[j]),
                        "distinct": bool(distinct[j])})
                top = distinct[order]
                wts = np.abs(sums[order])
                # permutation null: random TOPN subsets of all monomials
                rng = np.random.default_rng(int(r["seed"]) * 1000 + c)
                sims = np.array([distinct[rng.choice(len(sums), TOPN,
                                                     replace=False)].mean()
                                 for _ in range(9999)])
                obs = top.mean()
                null_rows.append({
                    "run": r["run"], "seed": r["seed"], "class": nm,
                    "degree": k,
                    "obs_count_share": float(obs),
                    "obs_weighted_share": float((wts * top).sum()
                                                / wts.sum()),
                    "null_uniform_multiset": comb(D, k) / comb(D + k - 1, k),
                    "pop_count_share": float(distinct.mean()),
                    "pop_weighted_share": float(
                        (np.abs(sums) * distinct).sum() / np.abs(sums).sum()),
                    "perm_p_lower": float(((sims <= obs).sum() + 1)
                                          / (len(sims) + 1)),
                    "perm_p_upper": float(((sims >= obs).sum() + 1)
                                          / (len(sims) + 1))})
        print(f"[coeff] {r['run']}", flush=True)
    top = pd.DataFrame(rows)
    top.to_csv(OUT / f"coeff_top{TOPN}_{stem}_K{a.coeff_K}.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / f"arity_null_{stem}_K{a.coeff_K}.csv",
                                   index=False)
    # stability of the top sets across seeds (mean pairwise Jaccard)
    st = []
    if len(top):
        for (nm, k), g in top.groupby(["class", "degree"]):
            sets = [set(gg["indices"]) for _, gg in g.groupby("seed")]
            jac = [len(s & t) / len(s | t)
                   for s, t in itertools.combinations(sets, 2)]
            st.append({"class": nm, "degree": k, "n_seeds": len(sets),
                       "jaccard_mean": float(np.mean(jac)) if jac else None,
                       "jaccard_min": float(np.min(jac)) if jac else None})
    pd.DataFrame(st).to_csv(OUT / f"coeff_stability_{stem}_K{a.coeff_K}.csv",
                            index=False)


# ------------------------------------------------------------------ perclass
def recall_at_fpr(score, pos, fpr):
    neg = score[~pos]
    if len(neg) == 0 or pos.sum() == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float((score[pos] > thr).mean())


def cmd_perclass(stem, a):
    from sklearn.metrics import average_precision_score, confusion_matrix
    d, meta = load_data(stem)
    x, y = d["x_test"], d["y_test"]
    classes = list(meta["label_mapping"])
    n_cls = meta["n_classes"]
    temporal = meta["dataset"] == "nfunsw"
    rows, cms, boots = [], [], []
    for r in runs_of(stem):
        if r["seed"] > a.max_seed:
            continue
        model = load_keras(r["run"])
        prob = model.predict(x, batch_size=8192, verbose=0)
        pred = prob.argmax(1)
        for c, nm in enumerate(classes):
            pos = y == c
            if not pos.any():
                continue
            rows.append({
                "run": r["run"], "variant": r["variant"], "K": r["K"],
                "seed": r["seed"], "class": nm, "n": int(pos.sum()),
                "precision": float((y[pred == c] == c).mean())
                if (pred == c).any() else 0.0,
                "recall": float((pred[pos] == c).mean()),
                "pr_auc": float(average_precision_score(pos, prob[:, c])),
                "recall_at_fpr_1e-3": recall_at_fpr(prob[:, c], pos, 1e-3),
                "recall_at_fpr_1e-2": recall_at_fpr(prob[:, c], pos, 1e-2)})
        cm = confusion_matrix(y, pred, labels=np.arange(n_cls))
        cms.append(pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                                columns=[f"pred_{c}" for c in classes])
                   .assign(run=r["run"]))
        if r["seed"] == 1:
            rng = np.random.default_rng(0)
            n = len(y)
            if temporal:   # contiguous blocks of the time-ordered test set
                nb = 200
                edges = np.linspace(0, n, nb + 1).astype(int)
                blocks = [np.arange(edges[i], edges[i + 1])
                          for i in range(nb)]
            vals = []
            for _ in range(a.n_boot):
                if temporal:
                    pick = rng.integers(0, nb, nb)
                    ii = np.concatenate([blocks[i] for i in pick])
                else:
                    ii = rng.integers(0, n, n)
                vals.append(macro_prauc(y[ii], prob[ii], n_cls))
            boots.append({"run": r["run"], "variant": r["variant"],
                          "K": r["K"],
                          "scheme": "block (200 time blocks)" if temporal
                          else "flow", "n_boot": a.n_boot,
                          "macro_prauc": macro_prauc(y, prob, n_cls),
                          "ci_lo": float(np.percentile(vals, 2.5)),
                          "ci_hi": float(np.percentile(vals, 97.5))})
        print(f"[perclass] {r['run']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"perclass_{stem}.csv", index=False)
    if cms:
        pd.concat(cms).to_csv(OUT / f"confusion_{stem}.csv")
    pd.DataFrame(boots).to_csv(OUT / f"bootstrap_{stem}.csv", index=False)


# ------------------------------------------------------------------ perturb
def top_features(w, c, k_feat, degree=2):
    """Distinct features of the largest |coefficient| monomials (degree 2)."""
    ti = [i for i, t in enumerate(w.terms) if t["degree"] == degree][0]
    uk, sums, idx = fold_monomials(w.coefficient_tensor(ti, c), degree)
    feats = []
    for j in np.argsort(-np.abs(sums)):
        for f in idx[j]:
            if f not in feats:
                feats.append(int(f))
        if len(feats) >= k_feat:
            return feats[:k_feat]
    return feats


def replace_cols(xc, cols, donor_pool, rng, mode, strata=None,
                 strata_pool=None):
    """Replace columns of xc by values from donor rows.

    marginal : each column from an independent random donor (v1 behaviour)
    joint    : all columns from one donor row (keeps their dependency)
    strat    : joint, donor drawn from the same stratum of an unselected
               covariate (dependency with the rest of the flow preserved)
    """
    out = xc.copy()
    n = len(xc)
    if mode == "marginal":
        for j in cols:
            out[:, j] = donor_pool[rng.integers(0, len(donor_pool), n), j]
    elif mode == "joint":
        rows = rng.integers(0, len(donor_pool), n)
        out[:, cols] = donor_pool[rows][:, cols]
    else:
        rows = np.empty(n, int)
        for s in np.unique(strata):
            m = strata == s
            pool = np.flatnonzero(strata_pool == s)
            if len(pool) == 0:
                pool = np.arange(len(donor_pool))
            rows[m] = pool[rng.integers(0, len(pool), m.sum())]
        out[:, cols] = donor_pool[rows][:, cols]
    return out


def cmd_perturb(stem, a):
    d, meta = load_data(stem)
    x, y = d["x_test"], d["y_test"]
    classes = list(meta["label_mapping"])
    D = x.shape[1]
    rng0 = np.random.default_rng(0)
    pool_idx = rng0.choice(len(x), size=min(200_000, len(x)), replace=False)
    pool = x[pool_idx]
    rows = []
    for r in runs_of(stem, "m4", 4):
        if r["seed"] > a.perturb_seeds:
            continue
        w = Weights(load_keras(r["run"]))

        def score(xx, c):
            lg = w.phis(xx).sum(0) + w.b
            pred = lg.argmax(1)
            other = lg.copy()
            other[:, c] = -np.inf
            return pred, lg[:, c] - other.max(1)

        for c, nm in enumerate(classes):
            ci = np.flatnonzero(y == c)
            if len(ci) == 0:
                continue
            if len(ci) > 20_000:
                ci = rng0.choice(ci, 20_000, replace=False)
            xc = x[ci]
            pred0, mar0 = score(xc, c)
            rec0 = float((pred0 == c).mean())
            for k_feat in (3, 6):
                sel = top_features(w, c, k_feat)
                others = [j for j in range(D) if j not in sel]
                corr = [abs(np.corrcoef(pool[:, j], pool[:, sel].sum(1))[0, 1])
                        if pool[:, j].std() > 0 else 0 for j in others]
                cov = others[int(np.nanargmax(corr))]
                edges = np.unique(np.quantile(pool[:, cov],
                                              np.linspace(0, 1, 11)))
                st_x = np.digitize(xc[:, cov], edges[1:-1])
                st_p = np.digitize(pool[:, cov], edges[1:-1])
                for mode in ("marginal", "joint", "strat"):
                    def effect(cols, rng):
                        xx = replace_cols(xc, cols, pool, rng, mode,
                                          st_x, st_p)
                        p, mg = score(xx, c)
                        return (rec0 - float((p == c).mean()),
                                float((p != pred0).mean()),
                                float((mar0 - mg).mean()))
                    rng = np.random.default_rng(r["seed"] * 7919 + c)
                    obs = effect(sel, rng)
                    ctl = np.array([effect(sorted(rng.choice(
                        D, k_feat, replace=False).tolist()), rng)
                        for _ in range(N_CONTROLS)])
                    row = {"run": r["run"], "seed": r["seed"], "class": nm,
                           "k_features": k_feat, "mode": mode,
                           "features": " | ".join(
                               str(d["feature_names"][j]) for j in sel),
                           "recall_before": rec0}
                    for j, met in enumerate(("recall_drop", "flip_rate",
                                             "margin_drop")):
                        row[f"{met}_obs"] = obs[j]
                        row[f"{met}_ctl_mean"] = float(ctl[:, j].mean())
                        row[f"{met}_ctl_sd"] = float(ctl[:, j].std())
                        row[f"{met}_p"] = float(
                            ((ctl[:, j] >= obs[j]).sum() + 1)
                            / (N_CONTROLS + 1))
                    rows.append(row)
            print(f"[perturb] {r['run']} {nm}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        n_tests = df.groupby(["run", "k_features", "mode"]).size().max()
        for met in ("recall_drop", "flip_rate", "margin_drop"):
            df[f"{met}_p_bonf"] = np.minimum(1.0, df[f"{met}_p"] * n_tests)
    df.to_csv(OUT / f"perturb_{stem}.csv", index=False)


# ------------------------------------------------------------------ stats
def exact_signflip_p(diff):
    """Two-sided exact sign-flip permutation p-value of the mean."""
    diff = np.asarray(diff, float)
    n = len(diff)
    obs = abs(diff.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=n)))
    means = np.abs((signs * diff).mean(1))
    return float((means >= obs - 1e-15).mean())


def cmd_stats(stem, a):
    from scipy import stats
    recs = []
    for r in runs_of(stem):
        key = r["variant"] + (f"_K{r['K']}" if r["variant"] in
                              ("m4", "m4g", "m4m", "m4ms") else "")
        if "heads" in r:
            key += f"_H{r['heads']}"
        if "lam" in r:
            key += f"_L{r['lam']:g}"
        recs.append({"config": key, "seed": r["seed"],
                     "pr_auc_macro": r["pr_auc_macro"],
                     "pr_auc_micro": r["pr_auc_micro"],
                     "f1_macro": r["f1_macro"], "mcc": r["mcc"],
                     "params": r["params"],
                     "flops_per_flow": r.get("flops_per_flow"),
                     "epochs": r["epochs_run"],
                     "hit_max_epochs": r["hit_max_epochs"],
                     "train_min": r["train_s"] / 60,
                     "infer_us": r["inference_us_per_flow"],
                     "unseen_pr_auc_macro": r.get("unseen_only", {})
                     .get("pr_auc_macro")})
    df = pd.DataFrame(recs)
    if df.empty:
        return
    df.to_csv(OUT / f"runs_{stem}.csv", index=False)
    g = df.groupby("config")
    summ = g.mean(numeric_only=True).add_suffix("_mean").join(
        g.std(numeric_only=True).add_suffix("_sd"))
    summ["n_seeds"] = g.size()
    summ.reset_index().to_csv(OUT / f"summary_{stem}.csv", index=False)

    pairs = [("m4g_K4", "m4_K4"), ("m4m_K4", "m4_K4"),
             ("nn_mlp", "m4m_K4"), ("m2w_H3", "m4_K4"), ("m2w_H5", "m4_K6"),
             ("mlp_s", "m4_K4"), ("m4ms_K4_L0.001", "m4_K4"),
             ("m4ms_K4_L0.01", "m4_K4"), ("m4_K5", "m4_K2"),
             ("m4_K6", "m4_K5")]
    rows = []
    for A, B in pairs:
        if A not in g.groups or B not in g.groups:
            continue
        m = df[df.config == A].set_index("seed")["pr_auc_macro"].to_frame(
            "a").join(df[df.config == B].set_index("seed")["pr_auc_macro"]
                      .to_frame("b"), how="inner")
        if len(m) < 2:
            continue
        diff = (m.a - m.b).to_numpy()
        t = stats.ttest_rel(m.a, m.b)
        rows.append({"A": A, "B": B, "n_pairs": len(m),
                     "delta": float(diff.mean()),
                     "dz": float(diff.mean() / diff.std(ddof=1))
                     if diff.std(ddof=1) > 0 else np.nan,
                     "p_t": float(t.pvalue),
                     "p_perm_exact": exact_signflip_p(diff)})
    st = pd.DataFrame(rows)
    if len(st):
        st["p_t_bonf"] = np.minimum(1, st["p_t"] * len(st))
        st["p_perm_bonf"] = np.minimum(1, st["p_perm_exact"] * len(st))
    st.to_csv(OUT / f"paired_{stem}.csv", index=False)

    # frontier: baselines with retraining stability
    br = [json.loads(p.read_text("utf-8"))
          for p in sorted(BASE.glob(f"{stem}_*.json"))]
    fr = []
    for model, grp in itertools.groupby(sorted(br, key=lambda r: r["model"]),
                                        key=lambda r: r["model"]):
        grp = list(grp)
        sets = [set(r["top10_terms"]) for r in grp]
        jac = [len(s & t) / len(s | t)
               for s, t in itertools.combinations(sets, 2)]
        fr.append({"model": model, "n_seeds": len(grp),
                   "pr_auc_macro_mean": np.mean([r["pr_auc_macro"]
                                                 for r in grp]),
                   "pr_auc_macro_sd": np.std([r["pr_auc_macro"]
                                              for r in grp], ddof=1)
                   if len(grp) > 1 else np.nan,
                   "n_terms": np.mean([r["n_terms"] for r in grp]),
                   "n_interactions": np.mean([r["n_interactions"]
                                              for r in grp])
                   if all(r["n_interactions"] is not None for r in grp)
                   else np.nan,
                   "top10_share": np.mean([r["top10_share"] for r in grp]),
                   "top10_jaccard": float(np.mean(jac)) if jac else np.nan,
                   "params_or_nodes": np.mean([r["params_or_nodes"]
                                               for r in grp]),
                   "train_min": np.mean([r["train_s"] for r in grp]) / 60,
                   "infer_us": np.mean([r["inference_us_per_flow"]
                                        for r in grp])})
    pd.DataFrame(fr).to_csv(OUT / f"frontier_{stem}.csv", index=False)


# ------------------------------------------------------------------ main
CMDS = {"exact": cmd_exact, "shares": cmd_shares, "coeff": cmd_coeff,
        "perclass": cmd_perclass, "perturb": cmd_perturb, "stats": cmd_stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS) + ["all"])
    ap.add_argument("--dataset", choices=["ciciot2023", "nfunsw"],
                    required=True)
    ap.add_argument("--tag", default="p2r")
    ap.add_argument("--max-seed", type=int, default=10)
    ap.add_argument("--coef-rows", type=int, default=256,
                    help="rows for the coefficient-tensor check (exact)")
    ap.add_argument("--coeff-K", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--perturb-seeds", type=int, default=3)
    a = ap.parse_args()
    stem = f"{a.dataset}_{a.tag}"
    for name in (CMDS if a.cmd == "all" else [a.cmd]):
        print(f"\n===== {name} :: {stem} =====", flush=True)
        CMDS[name](stem, a)


if __name__ == "__main__":
    main()
