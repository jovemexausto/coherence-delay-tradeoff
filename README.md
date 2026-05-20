# Useful Memory under Drift: Temporal-Validity Horizons for Finite-Memory Distribution Tracking

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--0201--9308-a6ce39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-0201-9308)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20041349-blue.svg)](https://doi.org/10.5281/zenodo.20041349)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)

Under drift, retained evidence is both useful and dangerous: more memory reduces
variance, but it also accumulates temporal misalignment with the present.

The manuscript studies the temporal-validity horizon for finite-memory
distribution tracking under drift and the useful-memory region induced by that
horizon. Its central law balances a finite-sample term against a drift-induced
staleness term,

\[
\mathbb E\,d\!\left(\widehat P_t^{(n)},P_t\right)
\le C_K n^{-a} + C_S\zeta n^H,
\]

yielding the horizon scale

\[
n^*(a,H)\asymp (C_K/\zeta)^{1/(a+H)}.
\]

At any tolerance level $\delta>0$, this envelope also induces a useful-memory
region: the set of memory lengths whose tracking error remains within
$(1+\delta)$ times the optimum.

The theory closes in a tractable one-dimensional proof model, a structural
Gaussian lower bound, and a Gaussian location benchmark. A fixed-$\varepsilon$
Sinkhorn extension remains conjectural.

Dynamic regret and adaptive windowing are neighboring problems. The object here
is the temporal validity of retained evidence for distribution tracking under
drift.

## Core idea

- retained evidence under drift is governed by a temporal-validity horizon
- the horizon induces a useful-memory region of near-optimal memory lengths
- finite-sample behavior and temporal roughness jointly determine both the optimal scale and the relative shape of that region
- the lower bound shows that the finite-memory optimum is structural, not a tuning artifact
- the Gaussian location model provides a full benchmark theorem for the horizon law
- the fixed-$\varepsilon$ Sinkhorn extension is stated as a conjecture supported by structural results and empirical evidence
- temporal validity can fail before changepoint evidence becomes statistically visible

## Regeneration

To regenerate the paper artifacts:

```bash
uv run umh-generate-conceptual
uv run umh-generate-elec2
uv run umh-generate-gaussian
uv run umh-generate-roughness
uv run umh-generate-invalidity-gap
uv run umh-generate-finite-sample
tectonic main.tex
```

## Citation

If you cite the repository artifact, please use:

```bibtex
@software{parreira2026useful_memory_horizon,
  title = {Useful Memory under Drift: Temporal-Validity Horizons for Finite-Memory Distribution Tracking},
  author = {Parreira, Vinicius},
  month = may,
  year = {2026},
  doi = {10.5281/zenodo.20041349},
  url = {https://doi.org/10.5281/zenodo.20041349}
}
```
