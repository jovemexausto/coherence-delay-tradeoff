"""Airlines downstream benchmark entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..airlines.artifacts import save_airlines_figure
from ..airlines.model import AirlinesConfig, run_airlines_benchmark
from ..airlines.reports import build_airlines_rows
from ..core.common import export_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/airlines"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("./artifacts/airlines")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_airlines_benchmark(
        AirlinesConfig(cache_path=args.artifacts_dir / "airlines.arff")
    )
    figure_path = args.figures_dir / "fig_airlines.pdf"
    csv_path = args.artifacts_dir / "airlines_summary.csv"
    save_airlines_figure(result, figure_path)
    export_summary_csv(build_airlines_rows(result), csv_path)
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
