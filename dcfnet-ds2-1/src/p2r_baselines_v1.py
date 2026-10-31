"""p2r_baselines_v1.py -- accuracy-auditability frontier baselines (R2-7, R1-12).

Trained on the same partitions as src.p2r_train_v2 (train only; val unused
except by the models' own early stopping) and evaluated on the same test set.

Models
  lr            multinomial logistic regression, lbfgs           (v1 Table 5)
  lr_l1_C{c}    sparse logistic regression, L1, saga, C = c
  poly2_l1      degree-2 polynomial logistic model, L1, SGD (all D(D+3)/2
                terms streamed in batches; standardized on the training set)
  dt_d{d}       decision tree, max_depth = d
  hgb           HistGradientBoosting, v1 settings                (v1 Table 5)
  hgb_small     HistGradientBoosting, 8 leaves, depth 3, 100 iterations
  rf            random forest, v1 settings                       (v1 Table 5)
  rf_small      random forest, 20 trees, depth 6
  ebm           Explainable Boosting Machine (package `interpret`), 10 pairs

Auditability measures written for every model
  n_terms           active terms (non-zero coefficients, leaves, EBM terms)
  n_interactions    active terms involving two or more features
  top10_share       share of total term importance carried by the 10 largest
  top10_terms       the 10 largest terms (used for the stability measure)
  train_s, inference_us_per_flow, params_or_nodes
Stability across retraining is the mean pairwise Jaccard index of top10_terms
over seeds, computed by src.p2r_analysis_v1.

Run:
  python -m src.p2r_baselines_v1 --dataset ciciot2023 --tag p2r --seeds 3
  python -m src.p2r_baselines_v1 --dataset nfunsw --tag p2r --models lr dt_d4
"""

import argparse
import itertools
import json
import time

import numpy as np

from src import config as C

OUT = C.OUT_DIR / "p2r" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)
ALL = ["lr", "lr_l1_C0.01", "lr_l1_C0.1", "poly2_l1", "dt_d4", "dt_d8",
       "hgb_small", "hgb", "rf_small", "rf", "ebm"]
DETERMINISTIC = {"lr", "lr_l1_C0.01", "lr_l1_C0.1"}


def sample_weight(y):
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.unique(y)
    w = compute_class_weight("balanced", classes=cls, y=y)
    lut = np.zeros(int(cls.max()) + 1)
    lut[cls] = w
    return lut[y]


class Poly2:
    """Degree-2 polynomial features x_i, x_i x_j (i <= j), standardized."""

    def __init__(self, D, names):
        self.pairs = [(i, j) for i in range(D) for j in range(i, D)]
        self.names = list(names) + [f"{names[i]}*{names[j]}"
                                    for i, j in self.pairs]
        self.arity = [1] * D + [1 if i == j else 2 for i, j in self.pairs]
        self.I = np.array([p[0] for p in self.pairs])
        self.J = np.array([p[1] for p in self.pairs])
        self.mu = self.sd = None

    def raw(self, x):
        return np.hstack([x, x[:, self.I] * x[:, self.J]]).astype("float32")

    def fit(self, x, rows=200_000, seed=0):
        idx = np.random.default_rng(seed).choice(
            len(x), size=min(rows, len(x)), replace=False)
        z = self.raw(x[idx])
        self.mu, sd = z.mean(0), z.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def __call__(self, x):
        return (self.raw(x) - self.mu) / self.sd


def fit_poly2(x, y, seed, epochs=5, batch=50_000):
    from sklearn.linear_model import SGDClassifier
    pf = Poly2(x.shape[1], [f"f{j}" for j in range(x.shape[1])]).fit(x)
    clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=1e-5,
                        learning_rate="optimal", random_state=seed)
    w = sample_weight(y)
    cls = np.unique(y)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = order[s:s + batch]
            clf.partial_fit(pf(x[b]), y[b], classes=cls, sample_weight=w[b])
    return pf, clf


def predict_proba(model, x, kind, batch=100_000):
    out = []
    for s in range(0, len(x), batch):
        xb = x[s:s + batch]
        if kind == "poly2_l1":
            pf, clf = model
            out.append(clf.predict_proba(pf(xb)))
        else:
            out.append(model.predict_proba(xb))
    return np.vstack(out)


def build(kind, seed):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    if kind == "lr":
        return LogisticRegression(solver="lbfgs", max_iter=2000,
                                  class_weight="balanced")
    if kind.startswith("lr_l1_C"):
        return LogisticRegression(penalty="l1", solver="saga",
                                  C=float(kind[7:]), max_iter=300, tol=1e-3,
                                  class_weight="balanced", random_state=seed)
    if kind.startswith("dt_d"):
        return DecisionTreeClassifier(max_depth=int(kind[4:]),
                                      class_weight="balanced",
                                      random_state=seed)
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            class_weight="balanced", random_state=seed)
    if kind == "hgb_small":
        return HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=8, max_depth=3, early_stopping=True,
            validation_fraction=0.1, class_weight="balanced",
            random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if kind == "rf_small":
        return RandomForestClassifier(
            n_estimators=20, max_depth=6, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    if kind == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(interactions=10, n_jobs=-1,
                                             random_state=seed)
    raise ValueError(kind)


def tree_paths_interactions(est):
    """Leaves whose root-to-leaf path uses >= 2 distinct features."""
    t = est.tree_
    n_int, stack = 0, [(0, frozenset())]
    while stack:
        node, feats = stack.pop()
        if t.children_left[node] == -1:
            n_int += len(feats) >= 2
            continue
        f = feats | {int(t.feature[node])}
        stack += [(t.children_left[node], f), (t.children_right[node], f)]
    return n_int


def audit(kind, model, names, x_s, y_s, seed):
    """Term counts, interaction counts and importance concentration."""
    from sklearn.inspection import permutation_importance
    a = {}
    if kind == "lr" or kind.startswith("lr_l1"):
        coef = model.coef_
        imp = np.abs(coef * x_s.std(0)).sum(0)
        a["n_terms"] = int((np.abs(coef) > 0).any(0).sum())
        a["n_interactions"] = 0
        a["params_or_nodes"] = int(coef.size + model.intercept_.size)
        terms = list(names)
    elif kind == "poly2_l1":
        pf, clf = model
        coef = clf.coef_
        z = pf(x_s)
        imp = np.abs(coef * z.std(0)).sum(0)
        act = (np.abs(coef) > 0).any(0)
        terms = list(names) + [f"{names[i]}*{names[j]}" for i, j in pf.pairs]
        a["n_terms"] = int(act.sum())
        a["n_interactions"] = int(sum(act[k] for k in range(len(act))
                                      if pf.arity[k] == 2))
        a["params_or_nodes"] = int(coef.size + clf.intercept_.size)
    elif kind.startswith("dt_d") or kind.startswith("rf"):
        ests = [model] if kind.startswith("dt_d") else model.estimators_
        a["n_terms"] = int(sum(e.get_n_leaves() for e in ests))
        a["n_interactions"] = int(sum(tree_paths_interactions(e)
                                      for e in ests))
        a["params_or_nodes"] = int(sum(e.tree_.node_count for e in ests))
        imp, terms = model.feature_importances_, list(names)
    elif kind.startswith("hgb"):
        leaves = sum(p.get_n_leaf_nodes() for it in model._predictors
                     for p in it)
        a["n_terms"] = int(leaves)
        a["n_interactions"] = None   # not exposed by scikit-learn predictors
        a["params_or_nodes"] = int(sum(p.nodes.shape[0]
                                       for it in model._predictors
                                       for p in it))
        r = permutation_importance(model, x_s, y_s, n_repeats=3,
                                   random_state=seed, n_jobs=1,
                                   scoring="f1_macro")
        imp, terms = np.clip(r.importances_mean, 0, None), list(names)
    elif kind == "ebm":
        terms = [str(t) for t in model.term_names_]
        imp = np.asarray(model.term_importances())
        a["n_terms"] = len(terms)
        a["n_interactions"] = int(sum(len(f) >= 2
                                      for f in model.term_features_))
        a["params_or_nodes"] = int(sum(np.asarray(s).size
                                       for s in model.term_scores_))
    order = np.argsort(-imp)
    tot = float(imp.sum())
    a["top10_share"] = float(imp[order[:10]].sum() / tot) if tot > 0 else None
    a["top10_terms"] = [terms[k] for k in order[:10]]
    return a


def metrics(prob, y, n_cls, classes):
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(y, classes=np.arange(n_cls))
    pred = prob.argmax(1)
    per = {}
    for c, nm in enumerate(classes):
        m = y == c
        if m.any():
            npred = int((pred == c).sum())
            tp = int(((pred == c) & m).sum())
            per[nm] = {"n": int(m.sum()), "recall": tp / int(m.sum()),
                       "precision": tp / npred if npred else 0.0,
                       "pr_auc": float(average_precision_score(
                           yb[:, c], prob[:, c]))}
    return {"pr_auc_macro": float(average_precision_score(yb, prob,
                                                          average="macro")),
            "pr_auc_micro": float(average_precision_score(yb, prob,
                                                          average="micro")),
            "f1_macro": float(f1_score(y, pred, average="macro",
                                       zero_division=0)),
            "per_class": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ciciot2023", "nfunsw"],
                    required=True)
    ap.add_argument("--tag", default="p2r")
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    stem = f"{a.dataset}_{a.tag}"
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads((C.PROCESSED_DIR / f"{stem}_meta.json")
                      .read_text("utf-8"))
    x_tr, y_tr = d["x_train"], d["y_train"]
    x_te, y_te = d["x_test"], d["y_test"]
    names = [str(n) for n in d["feature_names"]]
    n_cls, classes = meta["n_classes"], list(meta["label_mapping"])
    rng = np.random.default_rng(0)
    s_idx = rng.choice(len(y_te), size=min(20_000, len(y_te)), replace=False)
    x_s, y_s = x_te[s_idx], y_te[s_idx]

    for kind, seed in itertools.product(a.models, range(1, a.seeds + 1)):
        if kind in DETERMINISTIC and seed > 1:
            continue
        out = OUT / f"{stem}_{kind}_s{seed}.json"
        if out.exists():
            print(f"[skip] {out.name}", flush=True)
            continue
        print(f"\n===== {stem} / {kind} / seed {seed} =====", flush=True)
        try:
            t0 = time.perf_counter()
            if kind == "poly2_l1":
                model = fit_poly2(x_tr, y_tr, seed)
            else:
                model = build(kind, seed)
                if kind in ("hgb", "hgb_small"):
                    model.fit(x_tr, y_tr)
                else:
                    model.fit(x_tr, y_tr, sample_weight=sample_weight(y_tr)
                              if kind == "ebm" else None)
            t_train = time.perf_counter() - t0
        except ImportError as e:
            print(f"  [skip] {kind}: {e}", flush=True)
            continue
        t0 = time.perf_counter()
        prob = predict_proba(model, x_te, kind)
        t_inf = time.perf_counter() - t0
        res = metrics(prob, y_te, n_cls, classes)
        res.update(audit(kind, model, names, x_s, y_s, seed))
        res.update({"dataset": a.dataset, "tag": a.tag, "model": kind,
                    "seed": seed, "train_s": t_train,
                    "inference_us_per_flow": 1e6 * t_inf / len(y_te)})
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  macro PR-AUC {res['pr_auc_macro']:.4f}  terms "
              f"{res['n_terms']}  train {t_train / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
