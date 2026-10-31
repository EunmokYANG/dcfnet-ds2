"""p2r_baselines_tuned_v1.py -- tuned sparse linear baselines (R2-7).

The first pass of src.p2r_baselines_v1 left two linear baselines weak:
saga stopped at 300 iterations without converging on the unstandardized
signed-log inputs, and the degree-2 SGD model (5 epochs, alpha 1e-5) scored
below plain logistic regression. Both are refit here with their
hyperparameter chosen on the VALIDATION partition (the test set is not used
for any choice):

  lr_l1_tuned     z-score standardization fitted on the training rows, then
                  multinomial L1 logistic regression (saga, max_iter 2000,
                  tol 1e-4, balanced class weights), C in {0.001, 0.01, 0.1, 1}
  poly2_l1_tuned  all D(D+3)/2 degree-<=2 terms, standardized on the training
                  rows; L1 SGD log-loss with averaging (ASGD), 10 epochs,
                  balanced sample weights, alpha in {1e-6, 1e-5, 1e-4};
                  the chosen alpha is then fit with seeds 1-3

Each run writes outputs/p2r/baselines/{stem}_{model}_s{seed}.json with the
same fields as p2r_baselines_v1 plus the selection trace (validation macro
PR-AUC per value, iterations, convergence flag).

Run:
  python -m src.p2r_baselines_tuned_v1 --dataset ciciot2023 --tag p2r
  python -m src.p2r_baselines_tuned_v1 --dataset nfunsw --tag p2r
"""

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src import config as C
from src.p2r_baselines_v1 import OUT, Poly2, audit, metrics, sample_weight

C_GRID = [0.001, 0.01, 0.1, 1.0]
ALPHA_GRID = [1e-6, 1e-5, 1e-4]
SAGA_ITER = 2000
SGD_EPOCHS = 10


def zscore_fit(x):
    mu = x.mean(0)
    sd = x.std(0)
    return mu, np.where(sd > 0, sd, 1.0)


def predict_batched(fn, x, batch=100_000):
    return np.vstack([fn(x[s:s + batch]) for s in range(0, len(x), batch)])


def save(stem, name, seed, res):
    out = OUT / f"{stem}_{name}_s{seed}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  [save] {out.name}  test macro PR-AUC {res['pr_auc_macro']:.4f}",
          flush=True)


def run_lr_l1(stem, d, meta, names, x_s, y_s):
    from sklearn.linear_model import LogisticRegression
    if (OUT / f"{stem}_lr_l1_tuned_s1.json").exists():
        print("[skip] lr_l1_tuned", flush=True)
        return
    n_cls, classes = meta["n_classes"], list(meta["label_mapping"])
    mu, sd = zscore_fit(d["x_train"])
    z = lambda a: ((a - mu) / sd).astype("float32")
    xtr, xva, xte = z(d["x_train"]), z(d["x_val"]), z(d["x_test"])
    trace, best = [], None
    for c in C_GRID:
        print(f"\n===== {stem} / lr_l1_tuned / C={c:g} =====", flush=True)
        m = LogisticRegression(penalty="l1", solver="saga", C=c,
                               max_iter=SAGA_ITER, tol=1e-4,
                               class_weight="balanced", random_state=1)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(xtr, d["y_train"])
        t = time.perf_counter() - t0
        conv = not any(issubclass(w.category, ConvergenceWarning)
                       for w in wl)
        v = metrics(m.predict_proba(xva), d["y_val"], n_cls,
                    classes)["pr_auc_macro"]
        trace.append({"C": c, "val_pr_auc_macro": v,
                      "n_iter": int(np.max(m.n_iter_)), "converged": conv,
                      "train_s": t})
        print(f"  val macro PR-AUC {v:.4f}  iter {int(np.max(m.n_iter_))}  "
              f"converged {conv}  {t / 60:.1f} min", flush=True)
        if best is None or v > best[0]:
            best = (v, c, m, t)
    v, c, m, t = best
    t0 = time.perf_counter()
    prob = predict_batched(m.predict_proba, xte)
    t_inf = time.perf_counter() - t0
    res = metrics(prob, d["y_test"], n_cls, classes)
    res.update(audit("lr_l1_tuned", m, names, z(x_s), y_s, 1))
    res.update({"dataset": meta["dataset"], "tag": meta["tag"],
                "model": "lr_l1_tuned", "seed": 1, "selected_C": c,
                "selection": trace, "standardized": True,
                "train_s": t,
                "inference_us_per_flow": 1e6 * t_inf / len(prob)})
    save(stem, "lr_l1_tuned", 1, res)


def fit_poly2(pf, x, y, alpha, seed, batch=50_000):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=alpha,
                        average=True, random_state=seed)
    w = sample_weight(y)
    cls = np.unique(y)
    rng = np.random.default_rng(seed)
    for _ in range(SGD_EPOCHS):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = order[s:s + batch]
            clf.partial_fit(pf(x[b]), y[b], classes=cls, sample_weight=w[b])
    return clf


def run_poly2(stem, d, meta, names, x_s, y_s, seeds):
    n_cls, classes = meta["n_classes"], list(meta["label_mapping"])
    pf = Poly2(d["x_train"].shape[1], names).fit(d["x_train"])
    # kept outside the {stem}_*.json pattern read by p2r_analysis stats
    sel_path = OUT / f"selection_{stem}_poly2_l1_tuned.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text("utf-8"))
    else:
        trace = []
        for alpha in ALPHA_GRID:
            print(f"\n===== {stem} / poly2_l1_tuned / alpha={alpha:g} =====",
                  flush=True)
            t0 = time.perf_counter()
            clf = fit_poly2(pf, d["x_train"], d["y_train"], alpha, 1)
            t = time.perf_counter() - t0
            v = metrics(predict_batched(lambda a: clf.predict_proba(pf(a)),
                                        d["x_val"]),
                        d["y_val"], n_cls, classes)["pr_auc_macro"]
            trace.append({"alpha": alpha, "val_pr_auc_macro": v,
                          "train_s": t})
            print(f"  val macro PR-AUC {v:.4f}  {t / 60:.1f} min", flush=True)
        best = max(trace, key=lambda r: r["val_pr_auc_macro"])
        sel = {"selected_alpha": best["alpha"], "selection": trace}
        sel_path.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    alpha = sel["selected_alpha"]
    for seed in seeds:
        if (OUT / f"{stem}_poly2_l1_tuned_s{seed}.json").exists():
            print(f"[skip] poly2_l1_tuned s{seed}", flush=True)
            continue
        print(f"\n===== {stem} / poly2_l1_tuned / alpha={alpha:g} / "
              f"seed {seed} =====", flush=True)
        t0 = time.perf_counter()
        clf = fit_poly2(pf, d["x_train"], d["y_train"], alpha, seed)
        t = time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = predict_batched(lambda a: clf.predict_proba(pf(a)),
                               d["x_test"])
        t_inf = time.perf_counter() - t0
        res = metrics(prob, d["y_test"], n_cls, classes)
        res.update(audit("poly2_l1", (pf, clf), names, x_s, y_s, seed))
        res.update({"dataset": meta["dataset"], "tag": meta["tag"],
                    "model": "poly2_l1_tuned", "seed": seed,
                    "selected_alpha": alpha, "selection": sel["selection"],
                    "epochs": SGD_EPOCHS, "averaged": True, "train_s": t,
                    "inference_us_per_flow": 1e6 * t_inf / len(prob)})
        save(stem, "poly2_l1_tuned", seed, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ciciot2023", "nfunsw"],
                    required=True)
    ap.add_argument("--tag", default="p2r")
    ap.add_argument("--models", nargs="+",
                    default=["lr_l1_tuned", "poly2_l1_tuned"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    stem = f"{a.dataset}_{a.tag}"
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads((C.PROCESSED_DIR / f"{stem}_meta.json")
                      .read_text("utf-8"))
    names = [str(n) for n in d["feature_names"]]
    # same audit subsample as p2r_baselines_v1
    rng = np.random.default_rng(0)
    y_te = d["y_test"]
    s_idx = rng.choice(len(y_te), size=min(20_000, len(y_te)), replace=False)
    x_s, y_s = d["x_test"][s_idx], y_te[s_idx]
    if "lr_l1_tuned" in a.models:
        run_lr_l1(stem, d, meta, names, x_s, y_s)
    if "poly2_l1_tuned" in a.models:
        run_poly2(stem, d, meta, names, x_s, y_s, range(1, a.seeds + 1))


if __name__ == "__main__":
    main()
