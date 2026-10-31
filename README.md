# DCFNet-DS — exact polynomial-degree decomposition for network intrusion detection

Code and result tables for

> **Exact Polynomial-Degree Decomposition of a Network Intrusion Detector:
> Higher-Order Structure and the Reliability of Degree-Level Explanations**
> Eunmok Yang, Yujin Jeong, Changho Seo.
> Department of Convergence Science, Kongju National University.
> Submitted to *IEEE Access* (resubmission of Access-2026-42835).

The detector separates each class logit into an exact sum of homogeneous
polynomial terms, one per degree,

```
logit_c(x) = phi_1,c(x) + ... + phi_K,c(x) + b_c
```

## Releases

| Release | Content |
| --- | --- |
| **v2.0.0** (current) | Adds `dcfnet-ds2-1/`, the code and results of the revised manuscript. **Start here: [`dcfnet-ds2-1/README_p2r.md`](dcfnet-ds2-1/README_p2r.md).** |
| v1.0.0 | The original submission (top-level `src/`, `outputs/`, [`REPRODUCE_paper2_degree.md`](REPRODUCE_paper2_degree.md)). Kept unchanged for reference. |

**Correction.** In v1.0.0, `src/p2_coeff_attribution_v1.py` multiplied the
cross matrices in the wrong order for degree three and above. Degree-2
coefficients were correct; the degree-3 and degree-4 arity values of v1.0.0
are superseded by `dcfnet-ds2-1/src/p2r_analysis_v2.py`, whose `exact`
sub-command checks the coefficient tensors against the network output.

## What is and is not here

| | |
| --- | --- |
| Included | the model, the preprocessing and training code, the analysis and figure scripts, per-run results, and every result table the paper reports |
| Not included | the two datasets, which their authors distribute; the trained checkpoints and processed arrays, which the commands regenerate |

```
dcfnet-ds2-1/           revised manuscript (v2.0.0)
  src/                  data, training, baselines, analysis, figures
  run_*.bat             commands in order (Windows)
  outputs/p2r/          runs/, baselines/, analysis/, figures/, history/
src/, outputs/          original submission (v1.0.0)
```

## Citing

> E. Yang, Y. Jeong, and C. Seo, "DCFNet-DS: exact polynomial-degree
> decomposition for network intrusion detection," v2.0.0, Zenodo, 2026.

## License

MIT. See `LICENSE`.
