from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.core.common import export_summary_csv
from experiments.roughness_family import (
    RoughnessScalingConfig,
    build_misalignment_rows,
    build_optimal_window_rows,
    build_slope_rows,
    run_roughness_scaling_experiment,
    save_horizon_misalignment_figure,
    save_roughness_scaling_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roughness-indexed horizon scaling artifacts."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/roughness_family"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/roughness_family"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    result = run_roughness_scaling_experiment(RoughnessScalingConfig())
    save_roughness_scaling_figure(result, args.figures_dir / "fig_roughness_family.pdf")
    save_horizon_misalignment_figure(
        result, args.figures_dir / "fig_horizon_misalignment.pdf"
    )
    export_summary_csv(
        build_slope_rows(result), args.csv_dir / "roughness_family_slopes.csv"
    )
    export_summary_csv(
        build_optimal_window_rows(result),
        args.csv_dir / "roughness_family_optima.csv",
    )
    export_summary_csv(
        build_misalignment_rows(result),
        args.csv_dir / "roughness_family_misalignment.csv",
    )


if __name__ == "__main__":
    main()
