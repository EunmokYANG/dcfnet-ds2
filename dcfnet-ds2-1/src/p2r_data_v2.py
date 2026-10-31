"""p2r_data_v2.py -- leakage-resistant partitions for the paper-2 resubmission.

Addresses Access-2026-42835 comments R1-6, R1-7, R1-11, R1-13, R2-2, R2-6.

CICIoT2023 (data/source/CICIoT2023.csv, 34 subtype labels, merged CSV)
  * drop rows with inf/NaN (rule of the v1 pipeline)
  * map the 34 subtypes to 8 categories, keep the subtype code
  * remove exact duplicate feature vectors BEFORE splitting; a vector seen
    under more than one category is dropped entirely (label conflict)
  * stratified train/test (70/30), then a stratified validation set (20 % of
    train) inside train; --split-seed changes both draws (repeated partitions)
  The merged CSV has no device, session or capture identifier, so a
  group-disjoint split is impossible from this file. After deduplication no
  evaluation vector occurs in training.

NF-UNSW-NB15-v3 (data/source/NF-UNSW-NB15-v3.csv)
  * drop rows with inf/NaN, sort by FLOW_START_MILLISECONDS (stable)
  * earliest 70 % = train+val, latest 30 % = test; validation = the latest
    20 % of the train+val period, so model selection is forward-looking
  * timestamps, binary label, IPv4 addresses and ports are removed
  * exact duplicates are kept (a repeated vector later in time is a valid
    deployment observation); test rows whose vector occurs in train are
    flagged in `seen_test` for an unseen-only analysis

Input map (--scaler; fitted maps use the training rows only)
  signedlog (default)  sgn(x) log(1+|x|), no fitted parameters
  minmax, standard     affine maps of the raw features            (R2-2)
  signedlog_minmax     signed-log followed by min-max             (R2-2)
  signedlog_standard   signed-log followed by z-score (centring)  (R2-2)
--categorical encode (NF only): PROTOCOL and L7_PROTO one-hot (categories seen
in training; unseen codes map to all zeros) and the three TCP flag bitmasks
expanded to 8 binary bits each; these columns bypass the scaler    (R1-11)

Outputs (data/processed/): {dataset}_{tag}.npz and {dataset}_{tag}_meta.json

Run:
  python -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r
  python -m src.p2r_data_v2 --dataset nfunsw --tag p2r
  python -m src.p2r_data_v2 --dataset nfunsw --tag p2r_enc --categorical encode
  python -m src.p2r_data_v2 --dataset ciciot2023 --tag p2r_split2 --split-seed 2
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C

CHUNK = 200_000
VAL_FRAC = 0.2
TEST_FRAC = 0.3
SCALERS = ["signedlog", "minmax", "standard", "signedlog_minmax",
           "signedlog_standard"]


def log(msg, t0=[time.time()]):
    now = time.time()
    print(f"  [{now - t0[0]:7.1f}s] {msg}", flush=True)
    t0[0] = now


def read_clean(path):
    """Read a CSV in chunks, drop rows with inf/NaN, downcast float64."""
    parts, n_in = [], 0
    for ch in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        n_in += len(ch)
        ch = ch.replace([np.inf, -np.inf], np.nan).dropna()
        for c in ch.select_dtypes(include=["float64"]).columns:
            ch[c] = ch[c].astype("float32")
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    log(f"read {path.name}: {n_in:,} rows, {len(df):,} after inf/NaN removal")
    return df, n_in


def encode(labels, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[v] for v in labels], dtype="int64")


def counts(y, classes):
    return {classes[int(k)]: int(v)
            for k, v in zip(*np.unique(y, return_counts=True))}


def nonconstant(X, fit_rows):
    return [j for j in range(X.shape[1])
            if np.unique(X[fit_rows, j]).size > 1]


def apply_map(kind, X, fit_rows):
    """Apply the input map; fitted statistics come from fit_rows only."""
    X = X.astype("float64")
    if kind.startswith("signedlog"):
        X = np.sign(X) * np.log1p(np.abs(X))
    if kind in ("minmax", "signedlog_minmax"):
        lo, hi = X[fit_rows].min(0), X[fit_rows].max(0)
        X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    elif kind in ("standard", "signedlog_standard"):
        mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)
    return X.astype("float32")


def categorical_block(df, spec, fit_rows):
    """One-hot nominal codes (training vocabulary) and 8-bit flag expansion."""
    cols, names = [], []
    for c in spec["nominal_cols"]:
        v = df[c].to_numpy().astype("int64")
        vocab = np.unique(v[fit_rows])
        for code in vocab:
            cols.append((v == code).astype("float32"))
            names.append(f"{c}={code}")
    for c in spec["bitmask_cols"]:
        v = df[c].to_numpy().astype("int64")
        for b in range(8):
            cols.append(((v >> b) & 1).astype("float32"))
            names.append(f"{c}_b{b}")
    return np.stack(cols, axis=1), names


def finish(X, names, fit_rows, scaler, cat_block=None, cat_names=None):
    keep = nonconstant(X, fit_rows)
    removed = [names[j] for j in range(len(names)) if j not in set(keep)]
    Xs = apply_map(scaler, X[:, keep], fit_rows)
    out_names = [names[j] for j in keep]
    if cat_block is not None:
        ck = nonconstant(cat_block, fit_rows)
        Xs = np.hstack([Xs, cat_block[:, ck]])
        out_names += [cat_names[j] for j in ck]
        removed += [cat_names[j] for j in range(len(cat_names))
                    if j not in set(ck)]
    return Xs, out_names, removed


def build_cic(tag, split_seed, scaler):
    spec = C.DATASETS["ciciot2023"]
    classes, mapping = spec["classes"], C.CICIOT_CATEGORY_MAPPING
    df, n_in = read_clean(C.SOURCE_DIR / spec["raw_file"])
    lab = spec["label_col"]
    sub = df[lab].astype(str).to_numpy()
    unknown = sorted(set(sub) - set(mapping))
    if unknown:
        raise ValueError(f"unmapped CICIoT2023 labels: {unknown}")
    cat = np.array([mapping[s] for s in sub])
    feats = [c for c in df.columns if c != lab]
    X = df[feats].to_numpy(dtype="float32")
    del df
    n_clean = len(X)

    key = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    kdf = pd.DataFrame({"key": key, "cat": cat})
    n_cat = kdf.groupby("key")["cat"].nunique()
    conflict = kdf["key"].isin(set(n_cat.index[n_cat > 1])).to_numpy()
    first = ~kdf["key"].duplicated(keep="first").to_numpy()
    keep_rows = first & ~conflict
    n_dup, n_conf = int((~first & ~conflict).sum()), int(conflict.sum())
    log(f"dedup: {n_dup:,} duplicate rows removed, {int((n_cat > 1).sum()):,} "
        f"conflicting vectors ({n_conf:,} rows) removed, "
        f"{int(keep_rows.sum()):,} unique rows kept")
    X, cat, sub = X[keep_rows], cat[keep_rows], sub[keep_rows]
    y = encode(cat, classes)
    sub_names = sorted(mapping)
    sub_code = encode(sub, sub_names)

    idx = np.arange(len(y))
    tr_all, te = train_test_split(idx, test_size=TEST_FRAC,
                                  random_state=split_seed, stratify=y)
    tr, va = train_test_split(tr_all, test_size=VAL_FRAC,
                              random_state=split_seed, stratify=y[tr_all])
    log(f"split seed {split_seed}: train {len(tr):,} / val {len(va):,} / "
        f"test {len(te):,}")
    Xs, names, removed = finish(X, feats, tr, scaler)
    out = dict(x_train=Xs[tr], y_train=y[tr], x_val=Xs[va], y_val=y[va],
               x_test=Xs[te], y_test=y[te],
               sub_train=sub_code[tr], sub_val=sub_code[va],
               sub_test=sub_code[te],
               seen_test=np.zeros(len(te), dtype=bool),
               feature_names=np.array(names, dtype=object),
               subtype_names=np.array(sub_names, dtype=object))
    meta = {"dataset": "ciciot2023",
            "split": "stratified random after exact-duplicate removal",
            "split_seed": split_seed, "rows_read": n_in,
            "rows_after_nan": n_clean, "duplicate_rows_removed": n_dup,
            "conflicting_vectors": int((n_cat > 1).sum()),
            "conflicting_rows_removed": n_conf,
            "rows_unique": int(len(y)), "removed_constant": removed,
            "subtype_counts_unique": {k: int(v) for k, v in
                                      pd.Series(sub).value_counts().items()}}
    return out, meta, names, classes


def build_nf(tag, scaler, categorical):
    spec = C.DATASETS["nfunsw"]
    classes = spec["classes"]
    df, n_in = read_clean(C.SOURCE_DIR / spec["raw_file"])
    tcol, lab = spec["time_col"], spec["label_col"]
    df = df.iloc[np.argsort(df[tcol].to_numpy(), kind="stable")]
    df = df.reset_index(drop=True)
    t_start = df[tcol].to_numpy()
    y = encode(df[lab].astype(str).to_numpy(), classes)
    feats = [c for c in df.columns if c not in spec["drop_cols"] and c != lab]
    cat_cols = spec["nominal_cols"] + spec["bitmask_cols"]

    n = len(y)
    cut_te = int(n * (1 - TEST_FRAC))
    cut_va = int(cut_te * (1 - VAL_FRAC))
    tr, va, te = (np.arange(0, cut_va), np.arange(cut_va, cut_te),
                  np.arange(cut_te, n))
    log(f"chronological: train {len(tr):,} / val {len(va):,} / "
        f"test {len(te):,}")

    if categorical == "encode":
        num = [c for c in feats if c not in cat_cols]
        cb, cn = categorical_block(df, spec, tr)
        X = df[num].to_numpy(dtype="float32")
        del df
        Xs, names, removed = finish(X, num, tr, scaler, cb, cn)
    else:
        X = df[feats].to_numpy(dtype="float32")
        del df
        Xs, names, removed = finish(X, feats, tr, scaler)

    key = pd.util.hash_pandas_object(pd.DataFrame(Xs), index=False).to_numpy()
    train_keys = set(key[tr])
    seen = np.fromiter((k in train_keys for k in key[te]), dtype=bool,
                       count=len(te))
    log(f"test rows whose exact vector occurs in train: {int(seen.sum()):,} "
        f"({100 * seen.mean():.2f}%)")
    yte = y[te]
    out = dict(x_train=Xs[tr], y_train=y[tr], x_val=Xs[va], y_val=y[va],
               x_test=Xs[te], y_test=yte, seen_test=seen,
               feature_names=np.array(names, dtype=object))
    meta = {"dataset": "nfunsw", "split": "chronological train < val < test",
            "rows_read": n_in, "rows_after_nan": int(n),
            "time_bounds_ms": {
                "train": [int(t_start[tr[0]]), int(t_start[tr[-1]])],
                "val": [int(t_start[va[0]]), int(t_start[va[-1]])],
                "test": [int(t_start[te[0]]), int(t_start[te[-1]])]},
            "test_seen_in_train": int(seen.sum()),
            "test_seen_in_train_pct": float(100 * seen.mean()),
            "removed_constant": removed,
            "removed_identifier": spec["drop_cols"],
            "seen_test_by_class": {
                classes[c]: [int(seen[yte == c].sum()), int((yte == c).sum())]
                for c in np.unique(yte)}}
    return out, meta, names, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ciciot2023", "nfunsw"],
                    required=True)
    ap.add_argument("--tag", default="p2r")
    ap.add_argument("--split-seed", type=int, default=1,
                    help="CICIoT2023 only: seed of the train/val/test draw")
    ap.add_argument("--scaler", choices=SCALERS, default="signedlog")
    ap.add_argument("--categorical", choices=["keep", "encode"],
                    default="keep", help="NF only: R1-11 encoding")
    a = ap.parse_args()
    print(f"[source] {C.SOURCE_DIR}", flush=True)

    if a.dataset == "ciciot2023":
        if a.categorical != "keep":
            ap.error("--categorical encode applies to nfunsw only")
        out, meta, names, classes = build_cic(a.tag, a.split_seed, a.scaler)
    else:
        out, meta, names, classes = build_nf(a.tag, a.scaler, a.categorical)

    meta.update({
        "tag": a.tag, "protocol": "p2r_v2", "scaler": a.scaler,
        "categorical": a.categorical,
        "n_features": len(names), "feature_names": names,
        "n_classes": len(classes),
        "label_mapping": {c: i for i, c in enumerate(classes)},
        "class_counts_train": counts(out["y_train"], classes),
        "class_counts_val": counts(out["y_val"], classes),
        "class_counts_test": counts(out["y_test"], classes),
        "train_shape": list(out["x_train"].shape),
        "val_shape": list(out["x_val"].shape),
        "test_shape": list(out["x_test"].shape),
    })
    stem = f"{a.dataset}_{a.tag}"
    np.savez(C.PROCESSED_DIR / f"{stem}.npz", **out)
    (C.PROCESSED_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved {stem}.npz  features={len(names)}")
    for part in ("train", "val", "test"):
        print(f"    {part:5s} {meta[f'class_counts_{part}']}", flush=True)


if __name__ == "__main__":
    main()
