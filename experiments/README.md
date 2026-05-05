# Experiment Code

This directory contains the Python experiment code for the paper.

## Layout

- `run.py` - single public CLI with domain subcommands.
- `cli/` - thin command implementations.
- `particle/`, `gaussian/`, `bikes/`, `elec2/`, `kuairand/` - domain packages.
- `core/` - shared helpers.
- `artifacts/` - generated summaries.

## Usage

```bash
uv sync
uv run python run.py particle --experiment demo
uv run python run.py gaussian
uv run python run.py bikes
uv run python run.py elec2
uv run python run.py kuairand
uv run python run.py all
```

KuaiRand expects extracted data under `../data/kuairand/KuaiRand-Pure/data`.

## License

Code in this directory is licensed under `Apache-2.0`; see the repository root `LICENSE`.
