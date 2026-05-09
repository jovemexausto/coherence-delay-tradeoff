# Reproducibility

This repository currently supports the manuscript titled *The Coherence-Delay Trade-off: Temporal Geometry of Useful Memory Under Drift*.
The active evidence bundle is the Gaussian law validation plus the `cuberoot_adwin` synthetic benchmark suite.

## Setup

Prerequisites: `tectonic`, `uv`.

```bash
uv sync
```

## Active paper experiments

```bash
uv run python -m experiments.cli.run_gaussian
uv run python -m experiments.cli.run_cuberoot_adwin
```

These commands regenerate the figures and CSV artifacts used by the current manuscript.

## Full artifact refresh

```bash
uv run python -m experiments.cli.run_all
```

## Legacy / archival experiments

Earlier masking, KuaiRand, Bikes, and ELEC2 pipelines remain in the repository as archival artifacts. They are not the central evidence bundle for the current paper.

If you need the old KuaiRand data pipeline:

```bash
./scripts/fetch_kuairand.sh
```

## Manuscript

```bash
tectonic --outdir . main.tex
```
