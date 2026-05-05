# Tracking Experiments

This directory contains the **Python** code used to generate the figures and summary artifacts for the *Coherence-Delay Trade-off* paper.

## Project layout

- **`pyproject.toml`** - UV-managed project configuration.
- **`run_tgt.py`** - CLI entry point for the Gaussian tracker experiments.
- **`run_experiment.py`** - CLI entry point for the particle-tracker and benchmark experiments.
- **`tracking/`** - Gaussian tracker, particle tracker, ELEC2 and Bikes benchmarks, and KuaiRand evaluation code.
- **`artifacts/`** - generated CSV and Markdown summaries.
- **`../figures/`** - paper figure directory; the scripts write PDFs there directly.

## Current scope

The particle-tracker implementation uses a scalar bootstrap particle filter for a synthetic stream with:

- nonlinear latent dynamics;
- heavy-tailed observation noise (Student-t);
- action-coupled environment dynamics via an `influence` parameter;
- posterior tracking via particles;
- convergence diagnostics via normalized ESS and weight entropy;
- explicit actuation score `sigma_A`;
- ablation modes `full`, `fm1`, `fm2`, `fm3`;
- `action gap = |action - uncontrolled_state|` to show how far the controller is trying to pull the world;
- `coercive effort = influence * action_gap`, used by the effort-corrected score;
- effort-aware correction `sigma_P^E = sigma_P * exp(-lambda * effort / E0)`;
- empirical coherence and effort-corrected scores for coercive-masking diagnostics.

The synthetic tracker uses a simulator-side effort proxy based on the gap between action and latent state. That is appropriate for controlled experiments; for real datasets, it can later be replaced by an observable effort surrogate.

## Getting started

```bash
# From experiments/
uv sync
uv run python run_tgt.py
uv run python run_experiment.py
uv run python run_experiment.py --experiment ablation --output ../figures/fig_tpt_ablation.pdf
uv run python run_experiment.py --experiment masking --influence 0.3 --output ../figures/fig_tpt_masking.pdf
uv run python run_experiment.py --experiment masking-grid --output ../figures/fig_tpt_masking_grid.pdf
uv run python run_experiment.py --experiment active-benchmark --output ../figures/fig_tpt_active_benchmark.pdf
uv run python run_experiment.py --experiment kuairand-logged --output ../figures/fig_kuairand_active.pdf
```

The masking run writes:

- `artifacts/tpt_masking_summary.csv`
- `artifacts/tpt_masking_summary.md`

The masking-grid run writes:

- `artifacts/tpt_masking_grid_raw.csv`
- `artifacts/tpt_masking_grid_summary.csv`
- `artifacts/tpt_masking_grid_summary.md`

The active benchmark run writes:

- `artifacts/tpt_active_benchmark_summary.csv`
- `artifacts/tpt_active_benchmark_summary.md`

The KuaiRand logged benchmark expects `../data/kuairand/KuaiRand-Pure/data`.

## Interpretation notes

The `masking` experiment compares passive `FM-1` against coercive `FM-1` with positive influence. In that regime the stale model can keep the standard coherence score deceptively high by forcing the world toward the frozen action, while the effort-corrected score drops because the control effort rises.

If `influence=0`, the coercive effort is supposed to be zero. That is expected: there is no realized forcing of the world. The plots also show the raw `action gap` separately from realized coercive effort.

## License

Code in this directory is licensed under `Apache-2.0`; see the repository root `LICENSE`.
