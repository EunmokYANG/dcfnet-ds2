"""차수 게이트 분석 — 이 논문의 핵심 기여.

산출물
  outputs/tables/degree_alpha_{tag}.csv   공격 유형 x 차수 표
  outputs/tables/purity_{tag}.csv         동차성 검정 (차수 분리 증명)
  outputs/figures/degree_alpha_{tag}.png  누적 막대 그래프

실행:  python -m src.degree_analysis --dataset ciciot2023 --variant m3
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import models

from src import config as C
from src.model import CUSTOM_OBJECTS


def load_all(dataset, variant, tag=""):
    stem = f"{dataset}_{tag}" if tag else dataset
    d = np.load(C.PROCESSED_DIR / f"{stem}.npz", allow_pickle=True)
    meta = json.loads(
        (C.PROCESSED_DIR / f"{stem}_meta.json").read_text("utf-8")
    )
    path = C.MODEL_DIR / f"{stem}_{variant}{C.MODEL_EXT}"
    model = models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                              compile=False)
    return d, meta, model, stem


def is_ds(model):
    """선형 readout 구조(M4 계열)인지 판별."""
    return any(l.name == "logit_deg1" for l in model.layers)


def contribution_table(model, x, y, class_names, max_degree=C.MAX_DEGREE):
    """차수별 로짓 기여 phi_k = V_k u~_k 를 직접 계산한다.

    게이트 값이 아니라 실제 로짓 기여이므로
      sum_k phi_k + b = logit
    이 항등식으로 성립하며, 재매개변수화 여지가 없다.
    """
    names = [f"logit_deg{k}" for k in range(1, max_degree + 1)]
    if any(l.name == "logit_mlp" for l in model.layers):
        names.append("logit_mlp")
    sub = models.Model(model.input,
                       [model.get_layer(n).output for n in names])
    phis = sub.predict(x, batch_size=4096, verbose=0)
    if not isinstance(phis, list):
        phis = [phis]

    # 항등식 검증: sum_k phi_k + b == logit
    logit = models.Model(model.input, model.get_layer("logits").output) \
        .predict(x[:2048], batch_size=2048, verbose=0)
    recon = sum(p[:2048] for p in phis) + model.get_layer("logits").b.numpy()
    err = float(np.max(np.abs(recon - logit)))
    print(f"[항등식] max|sum_k phi_k + b - logit| = {err:.3e} "
          f"({'성립' if err < 1e-3 else '불일치'})")

    rows = []
    idx = np.arange(len(y))
    for c, nm in list(enumerate(class_names)) + [(None, "ALL")]:
        m = np.ones(len(y), bool) if c is None else (y == c)
        if m.sum() == 0:
            continue
        # 각 표본의 "정답 클래스" 로짓에 대한 기여만 본다.
        # ALL 행도 동일 기준이어야 개별 클래스와 비교 가능하다.
        sel = idx[m]
        imp = np.array([np.abs(p[sel, y[sel]]).mean() for p in phis])
        share = imp / (imp.sum() + 1e-12)
        row = {"Attack": nm, "N": int(m.sum())}
        row.update({f"I_{n.replace('logit_', '')}": float(v)
                    for n, v in zip(names, imp)})
        row.update({f"share_{n.replace('logit_', '')}": float(v)
                    for n, v in zip(names, share)})
        row["dominant"] = names[int(np.argmax(imp))].replace("logit_", "")
        deg_only = imp[:max_degree] / (imp[:max_degree].sum() + 1e-12)
        row["effective_degree"] = float(
            sum((k + 1) * deg_only[k] for k in range(max_degree)))
        rows.append(row)
    return pd.DataFrame(rows)


def alpha_table(model, x, y, class_names, max_degree=C.MAX_DEGREE):
    gate = models.Model(model.input,
                        model.get_layer("degree_alpha").output)
    a = gate.predict(x, batch_size=1024, verbose=0)

    rows = []
    for c, name in enumerate(class_names):
        m = y == c
        if m.sum() == 0:
            continue
        mean = a[m].mean(axis=0)
        row = {"Attack": name, "N": int(m.sum())}
        row.update({f"alpha_{k}": float(mean[k - 1])
                    for k in range(1, max_degree + 1)})
        row["dominant_degree"] = int(np.argmax(mean)) + 1
        row["effective_degree"] = float(
            sum(k * mean[k - 1] for k in range(1, max_degree + 1))
        )
        rows.append(row)

    mean = a.mean(axis=0)
    row = {"Attack": "ALL", "N": int(len(a))}
    row.update({f"alpha_{k}": float(mean[k - 1])
                for k in range(1, max_degree + 1)})
    row["dominant_degree"] = int(np.argmax(mean)) + 1
    row["effective_degree"] = float(
        sum(k * mean[k - 1] for k in range(1, max_degree + 1))
    )
    rows.append(row)
    return pd.DataFrame(rows)


def purity_check(model, input_dim, max_degree=C.MAX_DEGREE, c=2.0):
    """u_k(c*x) == c^k * u_k(x) 이면 u_k 는 정확히 k차 동차다항식."""
    rng = np.random.default_rng(C.SEED)
    x = rng.normal(size=(64, input_dim)).astype("float32")
    rows = []
    layer_names = {l.name for l in model.layers}
    # A gated model must be probed at gated_deg{k}: that is the term which
    # actually enters the logit sum. Testing scale_deg{k} would report the
    # cross stack, which the gate does not touch, and would wrongly suggest
    # that the gated contribution is still homogeneous.
    gated = "gated_deg2" in layer_names
    has_scale = "scale_deg2" in layer_names
    for k in range(2, max_degree + 1):
        if gated:
            lname = f"gated_deg{k}"
        else:
            lname = f"scale_deg{k}" if has_scale else f"cross_deg{k}"
        sub = models.Model(model.input, model.get_layer(lname).output)
        base = sub.predict(x, verbose=0)
        scaled = sub.predict(c * x, verbose=0)
        err = np.max(np.abs(scaled - (c ** k) * base)) / (
            np.max(np.abs(base)) + 1e-12
        )
        rows.append({"degree": k, "relative_error": float(err),
                     "is_homogeneous": bool(err < 1e-3)})
    return pd.DataFrame(rows)


def plot_alpha(df, tag, max_degree=C.MAX_DEGREE):
    plot_df = df[df["Attack"] != "ALL"]
    cols = [f"alpha_{k}" for k in range(1, max_degree + 1)]
    bottom = np.zeros(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, col in enumerate(cols, start=1):
        vals = plot_df[col].values
        ax.bar(plot_df["Attack"], vals, bottom=bottom,
               label=f"degree {k}")
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("gate weight $\\alpha_k$")
    ax.set_title(f"Interaction degree usage per attack type ({tag})")
    ax.legend(ncol=max_degree, fontsize=9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"degree_alpha_{tag}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--variant", default="m3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-degree", type=int, default=None)
    a = ap.parse_args()
    K = a.max_degree or C.MAX_DEGREE

    d, meta, model, stem = load_all(a.dataset, a.variant, a.tag)
    tag = f"{stem}_{a.variant}"
    names = C.DATASETS[a.dataset]["classes"]

    if is_ds(model):
        df = contribution_table(model, d["x_test"], d["y_test"], names, K)
        df.to_csv(C.TABLE_DIR / f"degree_contrib_{tag}.csv", index=False)
        cols = ["Attack", "N"] + [c for c in df.columns
                                  if c.startswith("share_")] + \
               ["dominant", "effective_degree"]
        print("\n[차수별 로짓 기여 비중]")
        print(df[cols].round(4).to_string(index=False))
    else:
        df = alpha_table(model, d["x_test"], d["y_test"], names)
        df.to_csv(C.TABLE_DIR / f"degree_alpha_{tag}.csv", index=False)
        print(df.to_string(index=False))

    pdf = purity_check(model, meta["n_features"], max_degree=K)
    pdf.to_csv(C.TABLE_DIR / f"purity_{tag}.csv", index=False)
    print("\n[동차성 검정]")
    print(pdf.to_string(index=False))

    if not is_ds(model):
        plot_alpha(df, tag)
        print(f"\n[save] outputs/figures/degree_alpha_{tag}.png")


if __name__ == "__main__":
    main()
