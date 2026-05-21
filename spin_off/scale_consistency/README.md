# Scale-Consistency Spin-Off

This workspace isolates the spin-off paper on structural inference for
power-law lag geometry under noisy transport estimates.

## Layout

- `scale_consistency.tex`: manuscript entrypoint
- `code/`: dedicated Python package for the spin-off
- `tests/`: dedicated spin-off tests
- `artifacts/`: figures, tables, and CSV outputs for this paper
- `notes/`: paper and experiment planning notes local to this workspace

## Local dependencies

The workspace carries its own `config/`, `bibliography.bib`, `pyproject.toml`,
and Python package under `code/scale_consistency/`.

## Immediate build command

Run from this directory:

`tectonic scale_consistency.tex`

## Local test command

With a workspace-local environment active:

`PYTHONPATH=code python -m unittest discover -s tests`

## V1 execution note

The active V1 paper-and-code plan is documented in
`notes/13-scale-consistency-v1-plan.md`.
