"""KuaiRand experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.common import export_summary_csv
from ..kuairand.artifacts import save_kuairand_figure
from ..kuairand.model import KuaiRandConfig, run_kuairand_active_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("../figures/kuairand"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=Path("./artifacts/kuairand")
    )
    parser.add_argument(
        "--kuairand-data-dir",
        type=Path,
        default=Path("../data/kuairand/KuaiRand-Pure/data"),
    )
    parser.add_argument("--kuairand-window-size", type=int, default=20)
    parser.add_argument("--kuairand-min-phase-count", type=int, default=20)
    parser.add_argument("--kuairand-max-users", type=int, default=1000)
    parser.add_argument("--kuairand-threshold-quantile", type=float, default=0.20)
    parser.add_argument("--kuairand-tcie-lambda", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_kuairand_active_benchmark(
            KuaiRandConfig(
                data_dir=args.kuairand_data_dir,
                window_size=args.kuairand_window_size,
                min_phase_count=args.kuairand_min_phase_count,
                max_users=args.kuairand_max_users,
                threshold_quantile=args.kuairand_threshold_quantile,
                tcie_lambda=args.kuairand_tcie_lambda,
            )
        )
    except FileNotFoundError as error:
        raise SystemExit(
            f"KuaiRand data not found at {args.kuairand_data_dir}."
        ) from error
    figure_path = args.figures_dir / "fig_kuairand.pdf"
    csv_path = args.artifacts_dir / "kuairand_summary.csv"
    save_kuairand_figure(result, figure_path)
    export_summary_csv(result.summary_rows, csv_path)
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
