# DCFNet-DS — exact polynomial-degree decomposition for network intrusion detection

Code and result tables for

> **Polynomial-Degree Decomposition of a Network Intrusion Detector: Exactness
> and the Accuracy–Auditability Trade-off**
> Eunmok Yang, Yujin Jeong, Changho Seo.
> Department of Convergence Science, Kongju National University.
> Submitted to *IEEE Access*.

The detector separates each class logit into an exact sum of homogeneous
polynomial terms, one per degree,

```
logit_c(x) = phi_1,c(x) + ... + phi_K,c(x) + b_c
```

which holds as an identity rather than as a post-hoc approximation, verified at
zero residual on 1,025,936 evaluation flows.

**Start here: [`REPRODUCE_paper2_degree.md`](REPRODUCE_paper2_degree.md).** It is
self-contained — datasets, environment, every command in order, and where each
table and figure of the paper comes from.

---

## What is and is not here

| | |
| --- | --- |
| Included | the model, the preprocessing and training code, the analysis scripts, and every result table the paper reports |
| Not included | the two datasets, which their authors distribute; the trained checkpoints, several hundred files that no analysis needs and that the commands regenerate |

This repository is a frozen snapshot for this paper. It carries only what the
paper needs: a reader is not asked to work out which files belong to other work.

```
src/                    the code, eighteen modules
data/                   empty; the guide says what to download and where
outputs/
  tables/               the experiment tables every analysis reads
  paper2_degree/
    tables/             the paper's tables
    figures/            the paper's figures
  models/               not tracked; the commands regenerate them
```

## Citing

> E. Yang, Y. Jeong, and C. Seo, "DCFNet-DS: exact polynomial-degree
> decomposition for network intrusion detection," v1.0.0, Zenodo, 2026.

The paper citation will be added once its DOI is assigned.

## License

MIT. See `LICENSE`.
