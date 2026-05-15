from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gaussian import (
    GaussianConfig,
    run_gaussian_ablation,
    run_sample_complexity_experiment,
    run_sinkhorn_runtime_experiment,
    run_ucurve_experiment,
)
from experiments.gaussian.artifacts import (
    save_ablation_figure,
    save_sigma_p_complexity_figure,
    save_sinkhorn_runtime_figure,
    save_ucurve_figure,
)
from experiments.gaussian.reports import (
    build_ablation_rows,
    build_sample_complexity_rows,
    build_sinkhorn_runtime_rows,
    build_ucurve_rows,
    export_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gaussian Paper 1 artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/gaussian"),
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/gaussian"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    ablation_results = run_gaussian_ablation(GaussianConfig(seed=args.seed))
    save_ablation_figure(
        ablation_results, args.figures_dir / "fig_gaussian_ablation.pdf"
    )
    export_rows_csv(
        build_ablation_rows(ablation_results),
        args.csv_dir / "gaussian_ablation_summary.csv",
    )

    ucurve_result = run_ucurve_experiment()
    save_ucurve_figure(ucurve_result, args.figures_dir / "fig_ucurve.pdf")
    export_rows_csv(
        build_ucurve_rows(ucurve_result), args.csv_dir / "gaussian_ucurve.csv"
    )

    sample_complexity_result = run_sample_complexity_experiment()
    save_sigma_p_complexity_figure(
        sample_complexity_result,
        args.figures_dir / "fig_gaussian_sinkhorn.pdf",
    )
    export_rows_csv(
        build_sample_complexity_rows(sample_complexity_result),
        args.csv_dir / "gaussian_sinkhorn.csv",
    )

    runtime_result = run_sinkhorn_runtime_experiment()
    save_sinkhorn_runtime_figure(
        runtime_result,
        args.figures_dir / "fig_gaussian_sinkhorn_runtime.pdf",
    )
    export_rows_csv(
        build_sinkhorn_runtime_rows(runtime_result),
        args.csv_dir / "gaussian_sinkhorn_runtime.csv",
    )


if __name__ == "__main__":
    main()
