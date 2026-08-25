# Reproducing paper 2 — polynomial-degree decomposition

> **Polynomial-Degree Decomposition of a Network Intrusion Detector: Exactness
> and the Accuracy–Auditability Trade-off.**
> Eunmok Yang, Yujin Jeong, Changho Seo. Submitted to *IEEE Access*.

This file is self-contained: everything needed to reproduce the paper is
here and in this repository.

The detector separates each class logit into an exact sum of homogeneous
polynomial terms, one per degree,

```
logit_c(x) = phi_1,c(x) + ... + phi_K,c(x) + b_c
```

which holds as an identity, verified at zero residual on 1,025,936 evaluation
flows. The experiment asks how far up the degree hierarchy detection reaches
and what exact decomposability costs against models that do not offer it.

| Variant | What it is |
| --- | --- |
| `m4` | the decomposable detector |
| `m4g` | the same with a learned degree gate |
| `m4m` | the same with an added dense branch, which is not decomposable |
| `nn_mlp` | a plain dense network |

---

## Datasets

Neither dataset is redistributed here. Download each from its source and place
the CSV in `data/source/`.

| Dataset | File expected | Source |
| --- | --- | --- |
| CICIoT2023 | `data/source/CICIoT2023.csv` | Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html |
| NF-UNSW-NB15-v3 | `data/source/NF-UNSW-NB15-v3.csv` | NetFlow v3 datasets, University of Queensland |

Both are used at full scale with their natural class distribution: 1,176,851
flows and 43 features on CICIoT2023, 2,242,931 flows and 47 features on
NF-UNSW-NB15-v3, after the identifier columns are dropped. The held-out
partitions are 353,056 and 672,880 flows, the 1,025,936 on which the identity
of Section III is checked.

Two files under `data/processed/` are in the repository rather than downloaded:
`ciciot2023_full_meta.json` and `nfunsw_full_meta.json`. Step 1 below rewrites
them, but they are committed so that the feature names behind the monomials of
Table VIII, the class order, the per-degree rms of the raw input and the class
counts of each split can be read without running anything. They contain no rows
of either dataset.

## Environment

TensorFlow 2.16.2 with Keras 3.15.1, CPU only. TensorFlow has shipped no
native Windows GPU build since 2.10, so the GPUs in this machine are not used;
at this model size — 6,939 parameters — the CPU run is not the bottleneck in
any case. The runs behind the paper were made on an Intel Core i9-7940X with
34 GB of memory under Windows 10; one training run takes roughly 15 to 20
minutes and stops after 68 to 103 epochs.

```bash
conda create -n dcfnet-ds python=3.11
conda activate dcfnet-ds
pip install -r requirements.txt
python -m src.check_env
```

`check_env.py` reports the versions it finds and stops if one is missing.

`src/config.py` holds every path and hyperparameter. Nothing else needs editing.

## Tags

Every experiment writes into shared tables keyed by a tag, and `src/sweep.py`
merges rows on (dataset, value, variant, tag) with `keep="last"`, so a run
under an existing tag replaces the rows already there. This paper writes under
`full` and `full_degree{K}`; keep `--base-tag full` on every sweep and the
committed rows stay intact.

---

## Commands

Run in order. Each stage writes into `outputs/` and later stages read what
earlier ones produced.

### 1. Data preparation and preprocessing

```bash
python -m src.prepare_data --dataset ciciot2023 --max-per-class 0
python -m src.prepare_data --dataset nfunsw     --max-per-class 0

python -m src.preprocess --dataset ciciot2023 --tag full --full --scaler signedlog --split random
python -m src.preprocess --dataset nfunsw     --tag full --full --scaler signedlog --split temporal
```

`--max-per-class 0` keeps the original distribution; the paper does not cap
classes. The scaling is held fixed at signed-log throughout.

### 2. Training and baselines

```bash
# variant comparison, five shared seeds
python -m src.multiseed --dataset ciciot2023 --tag full --variants m4 m4g m4m nn_mlp --seeds 5
python -m src.multiseed --dataset nfunsw     --tag full --variants m4 m4g m4m nn_mlp --seeds 5

# tree and linear baselines
python -m src.baselines --dataset ciciot2023 --tag full
python -m src.baselines --dataset nfunsw     --tag full

# degree sweep, three seeds
python -m src.sweep --kind degree --values 2 3 4 5 6 --variants m4 --seeds 3 --base-tag full
```

`multiseed` merges into `outputs/tables/seeds_{dataset}_full.csv`, so running
one variant later does not delete the rows of another. Add `--reuse` to
evaluate existing checkpoints instead of retraining.

### 3. Decomposition analysis

```bash
# per-degree contributions, identity and homogeneity checks
python -m src.degree_analysis --dataset ciciot2023 --variant m4  --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4  --tag full --max-degree 4
python -m src.degree_analysis --dataset ciciot2023 --variant m4g --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4g --tag full --max-degree 4
python -m src.degree_analysis --dataset ciciot2023 --variant m4m --tag full --max-degree 4
python -m src.degree_analysis --dataset nfunsw     --variant m4m --tag full --max-degree 4

# per-class shares at K = 6, which figure 6 reads
python -m src.degree_analysis --dataset ciciot2023 --variant m4 --tag full_degree6 --max-degree 6
python -m src.degree_analysis --dataset nfunsw     --variant m4 --tag full_degree6 --max-degree 6
```

### 4. Statistics, coefficients and controls

```bash
# paired significance tests
python -m src.p2_paired_stats_v1 --dataset ciciot2023 --tag full
python -m src.p2_paired_stats_v1 --dataset nfunsw     --tag full

# closed-form monomial coefficients
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/ciciot2023_full_m4.keras \
       --K 4 --features data/processed/ciciot2023_full_meta.json --topn 8
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/nfunsw_full_m4.keras \
       --K 4 --features data/processed/nfunsw_full_meta.json --topn 8 \
       --out outputs/paper2_degree/tables/p2_table6_coeff_attribution_nfunsw.csv

# interaction arity within each degree
python -m src.p2_arity_v1

# selectivity control for the recovered combinations
python -m src.perturbation --dataset ciciot2023 --tag full --variant m4 --n-feat 3 --n-control 30
python -m src.perturbation --dataset nfunsw     --tag full --variant m4 --n-feat 3 --n-control 30
```

### 5. Figures

```bash
python -m src.p2_figures_v2 --base-tag full
python -m src.p2_figures_v2 --base-tag full --only 4     # one figure only
```

---

## Where each output comes from

Tables go to `outputs/paper2_degree/tables/`, figures to
`outputs/paper2_degree/figures/`; the intermediate tables the analyses read are
in `outputs/tables/`.

| Item | Produced by | Output |
| --- | --- | --- |
| Table I — baseline settings | `src/config.py`, `src/baselines.py` | read from the source |
| Table II — exactness | `degree_analysis --variant m4` | `purity_{ds}_full_m4.csv`, identity residual printed |
| Table III — gated variant | `degree_analysis --variant m4g` | `purity_{ds}_full_m4g.csv` |
| Table IV — degree sweep | `sweep --kind degree` | `sweep_degree.csv` |
| Table V — detection performance | `multiseed`, `baselines` | `summary_{ds}_full.csv`, `metrics_baseline_{ds}_full.csv` |
| Table VI — ablation significance | `p2_paired_stats_v1` | `p2_paired_{ds}_full.csv` |
| Table VII — non-decomposable share | `degree_analysis --variant m4m` | `degree_contrib_{ds}_full_m4m.csv` |
| Table VIII — largest coefficients | `p2_coeff_attribution_v1` | `p2_table6_coeff_attribution*.csv` |
| Table IX — arity by degree | `p2_arity_v1` | `p2_table9_arity.csv` |
| Figures 1, 2 — architecture, pipeline | drawn, not computed | `figures/Figure1_architecture.svg`, `figures/Figure2_pipeline.svg` |
| Figure 3 — degree sweep | `p2_figures_v2 --only 1` | `p2_fig1_k_sweep.pdf` |
| Figure 4 — accuracy vs decomposability | `p2_figures_v2 --only 2` | `p2_fig2_tradeoff.pdf` |
| Figure 5 — arity by degree | `p2_arity_v1` | `p2_fig5_arity.pdf` |
| Figure 6 — class × degree shares | `p2_figures_v2 --only 4` | `p2_fig4_class_degree_heatmap.pdf` |

Figure and table numbering follows the paper; the script filenames keep the
numbering they had while the outputs were being produced.

## Parameter counts

```bash
python -m src.p2_coeff_attribution_v1 --ckpt outputs/models/ciciot2023_full_m4.keras --summary
```

Reports trainable, non-trainable and total counts. The closed form in the paper
is `(K-1)D^2 + KCD + C + 2K`, the last term being the running root-mean-square
and step counter each DegreeScale layer keeps.

---

## Two things worth knowing

**Seed counts differ by experiment.** The degree sweep uses three seeds and the
variant comparison uses five. They are separate runs, which is why their K = 4
entries differ slightly; neither difference exceeds one standard deviation.
Table IV and Table V of the paper state this.

**The validation split does not depend on the seed.** `train.py` draws it with
a fixed random state, so the seed changes weight initialization, batch ordering
and the non-determinism of parallel CPU reduction, but not the data partition.
Reported standard deviations therefore understate the variability that
resampling the split would add. Section IV-D of the paper says so.
