"""Temporal invalidity downstream benchmark entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.common import export_summary_csv
from ..temporal_invalidity.artifacts import save_temporal_invalidity_figure
from ..temporal_invalidity.model import (
    TemporalInvalidityConfig,
    run_temporal_invalidity_benchmark,
)
from ..temporal_invalidity.reports import build_temporal_invalidity_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("figures/temporal_invalidity")
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("./artifacts/temporal_invalidity")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_temporal_invalidity_benchmark(
        TemporalInvalidityConfig(
            seeds=(42, 43, 44, 45, 46, 47),
            # Match the paper's controlled drift recipe.
            phase_lengths=(2000, 2000, 2000),
            phase_zeta=(0.0005, 0.003, 0.0008),
        )
    )
    figure_path = args.figures_dir / "fig_temporal_invalidity.pdf"
    csv_path = args.artifacts_dir / "temporal_invalidity_summary.csv"
    save_temporal_invalidity_figure(result, figure_path)
    export_summary_csv(build_temporal_invalidity_rows(result), csv_path)
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
