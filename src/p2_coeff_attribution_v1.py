#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
p2_coeff_attribution_v1.py  --  Table VIII of the paper (feature-combination
attribution) plus the parameter counts of Table V.

Location : <project_root>/src/p2_coeff_attribution_v1.py
Run      : python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/<name>.keras --summary
           python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/<name>.keras --inspect
           python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/<name>.keras \
                  --K 4 --features data/processed/<ds>_meta.json --topn 8
Output   : outputs/paper2_degree/tables/p2_table6_coeff_attribution.csv

Reads the layer names that src/model.py::build_model_ds actually creates:
    cross_deg{k}_w   Dense(D, use_bias=False)   kernel is (D_in, D_out)
    scale_deg{k}     DegreeScale                weights: rms, step
    logit_deg{k}     Dense(C, use_bias=False)   kernel is (D, C)
    logits           BiasAdd                    weight: bias

Two transposes are required because Keras stores kernels as (in, out):
    the paper writes u_k = x * (W_k ubar_{k-1}) with W_k[j, i], while the
    layer computes sum_i kernel[i, j] * ubar_i, so W_k = kernel.T.
    The paper writes V_k as (C, D); the layer stores (D, C).

DegreeScale keeps a debiased moving average, so the divisor is
    sigma_k = rms_k / (1 - momentum ** max(step_k, 1)) + eps
and NOT the stored rms itself. Using the raw rms inflates every degree by
a different factor and destroys cross-degree comparability.

Section III-E closed form, with the divisor running to k (the readout is
applied to ubar_k, which already carries the division by sigma_k):
    c(i_1..i_k) = V_k[c, i_1] * prod_{m=2..k} W_m[i_{m-1}, i_m]
                  / prod_{m=1..k} sigma_m
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

# ============================================================================
# CONFIG -- layer name patterns from src/model.py; edit only if model.py changes
# ============================================================================
CFG = {
    "LAYER_W":     "cross_deg{k}_w",   # k = 2..K
    "LAYER_SCALE": "scale_deg{k}",     # k = 1..K
    "LAYER_V":     "logit_deg{k}",     # k = 1..K
    "LAYER_MLP":   "logit_mlp",        # present in variant m4m only
    "MOMENTUM_DEFAULT": 0.99,          # used only if the layer cannot report it
    # Enumerate all D**k monomial paths while this budget allows it.
    # Coefficients are exact only when the enumeration was exhaustive.
    "EXACT_MAX_PATHS": 5000000,
    "OUT_CSV": "outputs/paper2_degree/tables/p2_table6_coeff_attribution.csv",
}
# ============================================================================


def load_model(path):
    """Load a .keras file with the custom layers defined in src/model.py."""
    try:
        import keras
    except ImportError:                            # noqa: BLE001
        sys.exit("keras is required (TensorFlow 2.16 ships Keras 3).")
    try:
        from src.model import CUSTOM_OBJECTS
    except ImportError:
        sys.exit("run this as a module from the project root: "
                 "python -m src.p2_coeff_attribution_v1 ...")
    if not os.path.exists(path):
        sys.exit("checkpoint not found: %s" % path)
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                                   compile=False)


def summarise(model):
    """Report the parameter counts that Table V of the paper quotes."""
    model.summary()
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    n_ds = sum(1 for lay in model.layers
               if type(lay).__name__ == "DegreeScale")
    n_bn = sum(1 for lay in model.layers
               if type(lay).__name__ == "BatchNormalization")
    print("-" * 70)
    print("trainable     : %d" % tr)
    print("non-trainable : %d   (%d DegreeScale rms/step pairs, "
          "%d BatchNormalization moving statistics)" % (nt, n_ds, n_bn))
    print("total         : %d" % (tr + nt))
    print("Table V must quote this total. The closed-form count of "
          "Section III-F is (K-1)D^2 + KCD + C + 2K, where the final 2K "
          "is the non-trainable rms/step pair of each DegreeScale.")


def inspect(model):
    print("=" * 78)
    print("%-28s %-22s %s" % ("layer", "class", "weight shapes"))
    for lay in model.layers:
        shapes = [tuple(w.shape) for w in lay.weights]
        print("%-28s %-22s %s" % (lay.name, type(lay).__name__, shapes))
    print("=" * 78)
    print("Expected: cross_deg{k}_w, scale_deg{k}, logit_deg{k}, logits.")
    print("If the names differ, edit CFG at the top of this file.")


def sigma_of(layer):
    """Debiased DegreeScale divisor: rms / (1 - momentum ** max(step, 1)).

    Reads layer.s and layer.step directly. Variable naming differs between
    Keras versions, and a name lookup that silently misses would fall back
    to 1.0, leaving the coefficients unnormalised without any warning.
    """
    if not hasattr(layer, "s") or not hasattr(layer, "step"):
        sys.exit("layer '%s' has no s/step attributes; DegreeScale in "
                 "src/model.py must have changed." % layer.name)
    rms = float(np.asarray(layer.s).reshape(-1)[0])
    step = float(np.asarray(layer.step).reshape(-1)[0])
    mom = float(getattr(layer, "momentum", CFG["MOMENTUM_DEFAULT"]))
    eps = float(getattr(layer, "eps", 1e-6))
    if rms <= 0.0:
        sys.exit("layer '%s' has rms=%g: the moving average was never "
                 "updated, so sigma cannot be recovered." % (layer.name, rms))
    debias = 1.0 - mom ** max(step, 1.0)
    return rms / max(debias, eps) + eps


def build(model, K):
    """Return W (paper orientation), V (C, D) and sigma, all as float64."""
    W, V, sig = {}, {}, {}
    names = {lay.name: lay for lay in model.layers}

    for k in range(1, K + 1):
        nv = CFG["LAYER_V"].format(k=k)
        ns = CFG["LAYER_SCALE"].format(k=k)
        if nv not in names:
            sys.exit("layer '%s' not found. Is --K larger than the model's "
                     "max_degree? Run --inspect." % nv)
        V[k] = np.asarray(names[nv].weights[0], dtype=np.float64).T   # (C, D)
        if ns not in names:
            sys.exit("layer '%s' not found; sigma cannot be recovered and "
                     "coefficients would not be comparable across degrees."
                     % ns)
        sig[k] = sigma_of(names[ns])
        if k >= 2:
            nw = CFG["LAYER_W"].format(k=k)
            if nw not in names:
                sys.exit("layer '%s' not found. Run --inspect." % nw)
            W[k] = np.asarray(names[nw].weights[0], dtype=np.float64).T

    if CFG["LAYER_MLP"] in names:
        print("  [note] this checkpoint is variant m4m: a non-decomposable "
              "branch ('%s') carries part of the logit, so the coefficients "
              "below explain only the polynomial remainder."
              % CFG["LAYER_MLP"])
    print("  sigma: %s" % {k: round(v, 6) for k, v in sig.items()})
    return W, V, sig


def exact_monomials(W, V, sig, k, c, topn):
    """Full D**k enumeration folded onto unordered monomials, in numpy.

    Section III-E sums the coefficients of every ordered path producing the
    same monomial, so (a, b, c) and (b, a, c) are added together. Holding
    millions of paths as python tuples costs hundreds of megabytes; the
    same tensor is a few tens of megabytes here.

    Returns list of (sorted_index_tuple, summed_coefficient, n_paths).
    """
    D = V[k].shape[1]
    denom = float(np.prod([sig[m] for m in range(1, k + 1)]))

    coef = np.asarray(V[k][c], dtype=np.float64)          # over i_1
    for m in range(2, k + 1):
        coef = coef[..., None] * W[m]                     # broadcast one hop
    coef = (coef / denom).reshape(-1)

    idx = np.stack(np.indices((D,) * k, dtype=np.int32),
                   axis=-1).reshape(-1, k)
    idx.sort(axis=1)  # a monomial is a multiset

    # Fold each sorted index tuple into one base-D integer so that the
    # grouping is a 1-D unique instead of a lexsort over millions of rows.
    key = np.zeros(idx.shape[0], dtype=np.int64)
    for col in range(k):
        key = key * D + idx[:, col]
    uniq_key, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=coef, minlength=len(uniq_key))
    counts = np.bincount(inv, minlength=len(uniq_key))

    order = np.argsort(-np.abs(sums))[:topn]
    out = []
    for j in order:
        v, digits = int(uniq_key[j]), []
        for _ in range(k):
            digits.append(v % D)
            v //= D
        out.append((tuple(reversed(digits)), float(sums[j]), int(counts[j])))
    return out


def read_names(path, key, n, prefix, fallback_key=None):
    """Read a name list from a json file.

    Accepts three layouts, because preprocess.py stores the two lists
    differently: feature names are a plain list under 'feature_names',
    while class names live in 'label_mapping' as {name: index}.
      - a bare JSON list
      - obj[key] as a list
      - obj[fallback_key] as a {name: index} dict, ordered by index
    """
    if not path:
        return ["%s%d" % (prefix, i) for i in range(n)]
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    names = None
    if isinstance(obj, list):
        names = obj
    elif isinstance(obj.get(key), list):
        names = obj[key]
    elif fallback_key and isinstance(obj.get(fallback_key), dict):
        names = [k for k, _ in sorted(obj[fallback_key].items(),
                                      key=lambda kv: int(kv[1]))]

    if not names or len(names) != n:
        print("  [warn] %s: expected %d names for '%s'%s, got %s; "
              "falling back to indices"
              % (path, n, key,
                 "" if not fallback_key else " or '%s'" % fallback_key,
                 None if not names else len(names)))
        return ["%s%d" % (prefix, i) for i in range(n)]
    return [str(s) for s in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--features", default=None,
                    help="meta.json holding 'feature_names', or a bare list")
    ap.add_argument("--classes", default=None,
                    help="meta.json holding 'class_names', or a bare list")
    ap.add_argument("--summary", action="store_true",
                    help="print parameter counts for Table V and exit")
    ap.add_argument("--inspect", action="store_true",
                    help="print layer names and weight shapes and exit")
    ap.add_argument("--out", default=None, help="override the output CSV path")
    args = ap.parse_args()

    model = load_model(args.ckpt)
    if args.summary:
        summarise(model)
        return
    if args.inspect:
        inspect(model)
        return

    W, V, sig = build(model, args.K)
    C, D = V[1].shape
    fnames = read_names(args.features, "feature_names", D, "f")
    cnames = read_names(args.classes or args.features, "class_names", C,
                        "class", fallback_key="label_mapping")

    rows = []
    for c in range(C):
        for k in range(1, args.K + 1):
            n_all = D ** k
            if n_all > CFG["EXACT_MAX_PATHS"]:
                print("  [skip] class %s degree %d: %d paths exceed "
                      "EXACT_MAX_PATHS; raise it or report degrees <= %d only"
                      % (cnames[c], k, n_all, k - 1))
                continue
            for idxs, coef, n_paths in exact_monomials(W, V, sig, k, c,
                                                       args.topn):
                rows.append({
                    "class": cnames[c],
                    "degree": k,
                    "monomial": " x ".join(fnames[i] for i in idxs),
                    "indices": "-".join(str(i) for i in idxs),
                    "coefficient": coef,
                    "abs_coefficient": abs(coef),
                    "n_paths": n_paths,
                    "exact": True,
                })
    if not rows:
        sys.exit("nothing extracted; check --K and EXACT_MAX_PATHS")
    rows.sort(key=lambda r: (r["class"], r["degree"], -r["abs_coefficient"]))

    out_csv = args.out or CFG["OUT_CSV"]
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (out_csv, len(rows)))
    print("\nSection VI-D: this table describes the function the model "
          "fitted; it is not a causal claim.")


if __name__ == "__main__":
    main()