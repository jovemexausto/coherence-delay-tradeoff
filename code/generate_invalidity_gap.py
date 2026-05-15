from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.core.common import export_summary_csv
from experiments.invalidity_gap import (
    InvalidityGapConfig,
    build_gap_rows,
    build_trace_rows,
    run_invalidity_gap_experiment,
    save_invalidity_gap_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explicit invalidity-gap artifacts."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/invalidity_gap"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/invalidity_gap"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    result = run_invalidity_gap_experiment(InvalidityGapConfig())
    save_invalidity_gap_figure(result, args.figures_dir / "fig_invalidity_gap.pdf")
    export_summary_csv(
        build_gap_rows(result), args.csv_dir / "invalidity_gap_ablation.csv"
    )
    export_summary_csv(
        build_trace_rows(result), args.csv_dir / "invalidity_gap_traces.csv"
    )


if __name__ == "__main__":
    main()
