"""Bikes experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bikes.artifacts import save_bikes_figure
from ..bikes.model import BikesConfig, run_bikes_experiments
from ..bikes.reports import build_bikes_rows
from ..core.common import export_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("../figures/bikes"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("./artifacts/bikes"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bikes_experiments(BikesConfig())
    figure_path = args.figures_dir / "fig_bikes.pdf"
    csv_path = args.artifacts_dir / "bikes_summary.csv"
    save_bikes_figure(result, figure_path)
    export_summary_csv(build_bikes_rows(result), csv_path)
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
