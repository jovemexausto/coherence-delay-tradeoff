"""ELEC2 experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.common import export_summary_csv
from ..elec2.artifacts import (
    save_dynamic_nstar_figure,
    save_elec2_figure,
    save_umr_arena_figure,
)
from ..elec2.model import Elec2Config, run_elec2_experiments
from ..elec2.reports import build_elec2_arena_rows, build_elec2_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/elec2"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("./artifacts/elec2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_elec2_experiments(Elec2Config())
    figure_path = args.figures_dir / "fig_elec2.pdf"
    dynamic_path = args.figures_dir / "fig_dynamic_nstar.pdf"
    arena_figure_path = args.figures_dir / "fig_umr_arena.pdf"
    csv_path = args.artifacts_dir / "elec2_summary.csv"
    arena_csv_path = args.artifacts_dir / "elec2_arena_summary.csv"
    save_elec2_figure(result, figure_path)
    save_dynamic_nstar_figure(result, dynamic_path)
    save_umr_arena_figure(result, arena_figure_path)
    export_summary_csv(build_elec2_rows(result), csv_path)
    export_summary_csv(build_elec2_arena_rows(result), arena_csv_path)
    print(f"Saved {figure_path}")
    print(f"Saved {dynamic_path}")
    print(f"Saved {arena_figure_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {arena_csv_path}")


if __name__ == "__main__":
    main()
