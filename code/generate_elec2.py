from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.core.common import export_summary_csv
from experiments.elec2.artifacts import (
    save_dynamic_nstar_figure,
    save_elec2_figure,
    save_umr_arena_figure,
)
from experiments.elec2.model import Elec2Config, run_elec2_experiments
from experiments.elec2.reports import build_elec2_arena_rows, build_elec2_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ELEC2 Paper 1 artifacts.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/elec2"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/elec2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    result = run_elec2_experiments(Elec2Config())
    save_elec2_figure(result, args.figures_dir / "fig_elec2.pdf")
    save_dynamic_nstar_figure(result, args.figures_dir / "fig_dynamic_nstar.pdf")
    save_umr_arena_figure(result, args.figures_dir / "fig_umr_arena.pdf")
    export_summary_csv(build_elec2_rows(result), args.csv_dir / "elec2_summary.csv")
    export_summary_csv(
        build_elec2_arena_rows(result),
        args.csv_dir / "elec2_arena_summary.csv",
    )


if __name__ == "__main__":
    main()
