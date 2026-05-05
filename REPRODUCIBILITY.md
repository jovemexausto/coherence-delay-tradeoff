# Reproducibility

## Setup

Prerequisites: `tectonic`, `uv`.

```bash
uv sync
```

## KuaiRand data

```bash
./scripts/fetch_kuairand.sh
```

This populates `data/kuairand/KuaiRand-Pure/data`.

## Experiments

From `experiments/`:

```bash
uv run python run.py particle --experiment masking --output ../figures/particle/fig_particle_masking.pdf
uv run python run.py gaussian --figures-dir ../figures/gaussian
uv run python run.py bikes --figures-dir ../figures/bikes
uv run python run.py elec2 --figures-dir ../figures/elec2
uv run python run.py kuairand --figures-dir ../figures/kuairand
uv run python run.py all
```

## Manuscript

```bash
tectonic main.tex
```
