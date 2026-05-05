# Coherence-Delay Trade-off

Source repository for the preprint **"The Coherence-Delay Trade-off in Tracking Under Drift: A Cube-Root Law for Finite-Memory Estimation"**.

The project contains:

- the LaTeX source for the paper;
- the Python experiments used to generate the main figures and summary artifacts;
- public metadata for citation and reuse.

## Paper at a glance

The paper studies tracking under drift as a finite-memory estimation problem. When tracking error is decomposed into a statistical term and a drift-induced staleness term, the resulting trade-off yields a cube-root law for the optimal effective memory. In action-coupled settings, the repository also includes experiments for coherence scores, effort-corrected diagnostics, and coercive masking.

## Repository layout

- `main.tex`: top-level LaTeX entry point.
- `frontmatter/`, `theory/`, `discussion/`, `appendices/`, `config/`: manuscript source.
- `bibliography.bib`: BibTeX database.
- `figures/`: committed paper figures.
- `experiments/`: Python code used to regenerate figures and CSV/Markdown summaries.
- `CITATION.cff`: citation metadata for GitHub and archival workflows.

## Build the manuscript

Prerequisite: `tectonic`.

```bash
tectonic main.tex
```

This writes `main.pdf` in the repository root.

## Reproduce experiments

Prerequisite: `uv`.

```bash
cd experiments
uv sync
uv run python run.py particle --experiment masking --output ../figures/particle/fig_particle_masking.pdf
uv run python run.py gaussian --figures-dir ../figures/gaussian
uv run python run.py bikes --figures-dir ../figures/bikes
uv run python run.py elec2 --figures-dir ../figures/elec2
uv run python run.py kuairand --figures-dir ../figures/kuairand
uv run python run.py all
```

The experiment suite writes figures to `../figures/` and summary tables to `experiments/artifacts/`.

## Data-dependent runs

- The synthetic Gaussian and particle-tracker experiments run from the repository alone.
- The KuaiRand benchmark expects extracted data under `data/kuairand/KuaiRand-Pure/data`.

To fetch it:

```bash
./scripts/fetch_kuairand.sh
```

## Citation

Use the metadata in `CITATION.cff` or cite the repository URL listed in `appendices/code_repository.tex`.

## License

`LICENSE` defines the split licensing for this repository:

- manuscript source, documentation, and figures: `CC BY 4.0`;
- software under `experiments/`: `Apache-2.0`.
