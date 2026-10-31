# dcfnet-ds2-1 — code and results of the revised manuscript

*Exact Polynomial-Degree Decomposition of a Network Intrusion Detector:
Higher-Order Structure and the Reliability of Degree-Level Explanations*
(resubmission of IEEE Access Access-2026-42835).

This folder is self-contained. `src/model.py` and `src/config.py` are copies of
release v1.0.0 (`config.py` only adds a fallback path to
`../../dcfnet-ds/data/source`); all other scripts are new.

## Data
Put `CICIoT2023.csv` (merged CSV, 34 subtype labels) and `NF-UNSW-NB15-v3.csv`
in `data/source/`. Both datasets are distributed by their authors.

## Environment
`pip install -r ../requirements.txt`
(tensorflow 2.16.2, numpy 1.26.4, scikit-learn 1.9.0, interpret 0.7.8).
Do not let another package upgrade numpy to 2.x; TensorFlow 2.16.2 needs 1.26.

## Order of execution (Windows, conda env `dcfnet-ds`)
| Step | Batch file | What it does | Time (i9-7940X) |
|---|---|---|---|
| 1 | `run_1_data_v1.bat` | partitions (CIC: duplicates removed, stratified, split seeds 1–3; NF: chronological train < val < test), alternative input maps, one-hot/bit encoding | ~20 min |
| 2 | `run_2_train_v2.bat` | two windows (CIC, NF), 13 configurations × 10 seeds, then sensitivity runs (315 runs) | ~13 h |
| 3 | `run_3_baselines_v2.bat` | LR, L1-LR, poly-2 L1, decision trees, RF, HGB, EBM | several hours |
| 3b | `run_3b_baselines_tuned_v1.bat` | L1-LR and poly-2 L1 with penalty selected on validation | several hours |
| 4 | `run_4_analysis_v2.bat` | all analyses → `outputs/p2r/analysis/*.csv` | hours |
| 5 | `run_5_figures_v1.bat` | Figs. 3–7 → `outputs/p2r/figures/` | 1 min |

Every run writes `outputs/p2r/runs/<run>.json`; rerunning a batch file skips
finished runs. `run_2a/2b` start one dataset only; `run_3b_4_after3_v1.bat`
chains steps 3b and 4 after step 3.

## Scripts
| Script | Content | Paper |
|---|---|---|
| `src/p2r_data_v2.py` | partitions, deduplication, input maps, encodings | Sec. IV-A, Table 2 |
| `src/p2r_train_v2.py` | m4 (K = 2..6), m2w, mlp_s, m4g, m4m, m4ms, MLP | Tables 5, 6 |
| `src/p2r_baselines_v1.py`, `src/p2r_baselines_tuned_v1.py` | frontier baselines and audit measures | Table 15 |
| `src/p2r_analysis_v2.py` | `exact` (three-level exactness), `shares` (magnitude, centred, margin, removal), `coeff` (coefficients, stability, arity nulls), `perclass`, `perturb`, `stats` | Tables 4, 7–14 |
| `src/p2r_figures_v1.py` | figures | Figs. 3–7 |

Fig. 1 and Fig. 2 are TikZ drawings included with the manuscript source.
