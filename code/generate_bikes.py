from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bikes.artifacts import (
    save_bikes_figure,
    save_dynamic_nstar_figure,
    save_umr_arena_figure,
)
from experiments.bikes.model import BikesConfig, run_bikes_experiments
from experiments.bikes.reports import build_bikes_arena_rows, build_bikes_rows
from experiments.core.common import export_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bikes Paper 1 artifacts.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/bikes"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/bikes"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    result = run_bikes_experiments(BikesConfig())
    save_bikes_figure(result, args.figures_dir / "fig_bikes.pdf")
    save_dynamic_nstar_figure(result, args.figures_dir / "fig_dynamic_nstar.pdf")
    save_umr_arena_figure(result, args.figures_dir / "fig_umr_arena.pdf")
    export_summary_csv(build_bikes_rows(result), args.csv_dir / "bikes_summary.csv")
    export_summary_csv(
        build_bikes_arena_rows(result),
        args.csv_dir / "bikes_arena_summary.csv",
    )


if __name__ == "__main__":
    main()
