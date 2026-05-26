# Notebooks

This directory holds notebook-style analysis surfaces for the controller spin-off.

The initial notebook layout is:

- `controller_analysis.py`: notebook-style script (`# %%` cells) for
  reproducible loading, aggregation, CSV export, and plotting of controller
  benchmarks.

Run it with both package roots on `PYTHONPATH`, for example:

`PYTHONPATH=../code:../../scale-consistency/code uv run python notebooks/controller_analysis.py`

The script writes analysis-ready CSV files under
`artifacts/{csv,figures,tables}/controller_analysis/`.
