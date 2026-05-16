# Useful Memory Has a Horizon

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--0201--9308-a6ce39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-0201-9308)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20041349-blue.svg)](https://doi.org/10.5281/zenodo.20041349)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)

Under drift, retained evidence is both useful and dangerous: more memory reduces
variance, but it also accumulates temporal misalignment with the present.

The manuscript studies a carrier-roughness useful-memory horizon law, the
family of useful-memory scales it induces, the worst-case
Lipschitz cube-root regime as the $H=1$, $a=1/2$ case, a structural Gaussian
lower-bound witness, and the invalidity gap between temporal validity and
changepoint evidence.

This framing is adjacent to dynamic regret and adaptive windowing, but its
primary object is different: the temporal validity of retained evidence for
distribution tracking under drift.

## Core idea

- useful memory under drift is governed by a carrier-roughness useful-memory horizon law
- finite-sample carrier behavior and temporal roughness jointly determine the useful-memory scale
- the cube-root law is the `a=1/2, H=1` case of the carrier-roughness useful-memory horizon law
- the lower bound shows that the finite-memory optimum is structural, not a tuning artifact
- multiple carrier instantiations feed the same useful-memory horizon law
- temporal validity can fail before changepoint evidence becomes statistically visible

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
