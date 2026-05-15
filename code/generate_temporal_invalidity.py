from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.core.common import export_summary_csv
from experiments.temporal_invalidity.artifacts import save_temporal_invalidity_figure
from experiments.temporal_invalidity.model import (
    TemporalInvalidityConfig,
    run_temporal_invalidity_benchmark,
)
from experiments.temporal_invalidity.reports import build_temporal_invalidity_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled downstream Paper 1 artifacts."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/temporal_invalidity"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/temporal_invalidity"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    result = run_temporal_invalidity_benchmark(
        TemporalInvalidityConfig(
            seeds=(42, 43, 44, 45, 46, 47),
            phase_lengths=(2000, 2000, 2000),
            phase_zeta=(0.0005, 0.003, 0.0008),
        )
    )
    save_temporal_invalidity_figure(
        result,
        args.figures_dir / "fig_temporal_invalidity.pdf",
    )
    export_summary_csv(
        build_temporal_invalidity_rows(result),
        args.csv_dir / "temporal_invalidity_summary.csv",
    )


if __name__ == "__main__":
    main()
