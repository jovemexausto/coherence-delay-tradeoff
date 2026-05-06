# Coherence-Delay Trade-off

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--0201--9308-a6ce39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-0201-9308)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20041349-blue.svg)](https://doi.org/10.5281/zenodo.20041349)
[![License: CC BY 4.0 / Apache-2.0](https://img.shields.io/badge/License-CC%20BY%204.0%20%2F%20Apache--2.0-blue.svg)](LICENSE)

**The Coherence-Delay Trade-off and Coercive Masking Under Drift: Finite-Memory Tracking Limits and Effort-Aware Diagnostics**

Tracking under drift is a race against obsolescence: you want enough memory to be precise, but not so much that the estimate lags behind the world.

This work shows that, once those two forces are written explicitly, the optimal effective memory in the finite-memory averaging class follows a cube-root law. In action-coupled settings, the same lens also explains why standard coherence measures can look stable while effort-aware diagnostics reveal the cost of forcing the world to match a frozen action.

## Core idea

- finite-memory estimation under drift
- cube-root scaling for the optimal memory budget
- CI, CI^E, coercive masking, and the paper's tracking/diagnostic stack

## Reproducibility

See `REPRODUCIBILITY.md` for setup and run instructions.

## Citation

If you use this work, please cite:

```bibtex
@software{parreira2026coherence_delay_tradeoff,
  title = {The Coherence-Delay Trade-off and Coercive Masking Under Drift: Finite-Memory Tracking Limits and Effort-Aware Diagnostics},
  author = {Parreira, Vinicius},
  month = may,
  year = {2026},
  doi = {10.5281/zenodo.20041349},
  url = {https://doi.org/10.5281/zenodo.20041349}
}
```

## Title Transition

This repository's v2 manuscript is titled:

- `The Coherence-Delay Trade-off and Coercive Masking Under Drift: Finite-Memory Tracking Limits and Effort-Aware Diagnostics`

Earlier drafts circulated under the shorter title:

- `The Coherence-Delay Trade-off in Tracking Under Drift: A Cube-Root Law for Finite-Memory Estimation`

Both titles refer to the same project lineage. For new citations, use the v2 title above. See `TITLE_TRANSITION.md` for the transition note recorded alongside the repository metadata.
