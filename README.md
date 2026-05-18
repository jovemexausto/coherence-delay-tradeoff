# Useful Memory Has a Horizon

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--0201--9308-a6ce39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-0201-9308)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20041349-blue.svg)](https://doi.org/10.5281/zenodo.20041349)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)

Under drift, retained evidence is both useful and dangerous: more memory reduces
variance, but it also accumulates temporal misalignment with the present.

The manuscript studies the temporal-validity horizon for finite-memory
distribution tracking under drift. Its central law balances a finite-sample term
against a drift-induced staleness term,

\[
\mathbb E\,d\!\left(\widehat P_t^{(n)},P_t\right)
\le C_K n^{-a} + C_S\zeta n^H,
\]

yielding the horizon scale

\[
n^*(a,H)\asymp (C_K/\zeta)^{1/(a+H)}.
\]

The paper closes this theory in a tractable one-dimensional proof model,
establishes a structural Gaussian lower bound and a Gaussian location benchmark,
and states an explicit conjectural extension for fixed-$\varepsilon$ Sinkhorn
geometry.

This framing is adjacent to dynamic regret and adaptive windowing, but its
primary object is different: the temporal validity of retained evidence for
distribution tracking under drift.

## Core idea

- retained evidence under drift is governed by a temporal-validity horizon
- finite-sample behavior and temporal roughness jointly determine the optimal memory scale
- the lower bound shows that the finite-memory optimum is structural, not a tuning artifact
- the Gaussian location model provides a full benchmark theorem for the horizon law
- the fixed-$\varepsilon$ Sinkhorn extension is stated as a conjecture supported by structural results and empirical evidence
- temporal validity can fail before changepoint evidence becomes statistically visible

## Claim status

- `theorem`: abstract upper law, optimized horizon law, uniform-window staleness bound, one-dimensional root-$n$ proof model, structural Gaussian lower bound at the exponent level, and Gaussian location minimax benchmark
- `conjecture`: fixed-$\varepsilon$ Sinkhorn horizon inheritance and broader regular-family horizon inheritance
- `repository provenance`: calibration tables, artifact generation, and extended empirical evidence

## Repository layout

- `main.tex` and the section files contain the manuscript.
- `code/useful_memory_horizon/` contains the reproduction scripts for the figures, tables, and CSV outputs used in the paper.
- `artifacts/` contains the generated figures, tables, and CSVs used by the manuscript.
- `tests/` contains regression checks for the roughness-family and invalidity-gap pipelines.

## Regeneration

To regenerate the paper artifacts:

```bash
uv run umh-generate-conceptual
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
  title = {Useful Memory Has a Horizon},
  author = {Parreira, Vinicius},
  month = may,
  year = {2026},
  doi = {10.5281/zenodo.20041349},
  url = {https://doi.org/10.5281/zenodo.20041349}
}
```
