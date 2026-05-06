"""Gaussian experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..gaussian import (
    GaussianConfig,
    run_gaussian_ablation,
    run_sample_complexity_experiment,
    run_sinkhorn_runtime_experiment,
    run_ucurve_experiment,
)
from ..gaussian.artifacts import (
    save_ablation_figure,
    save_sigma_p_complexity_figure,
    save_sinkhorn_runtime_figure,
    save_ucurve_figure,
)
from ..gaussian.reports import (
    build_ablation_rows,
    build_sample_complexity_rows,
    build_sinkhorn_runtime_rows,
    build_ucurve_rows,
    export_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../figures/gaussian"),
        help="Directory where PDF figures will be written",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts/gaussian"),
        help="Directory for CSV summaries",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the ablation run"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir
    artifacts_dir = args.artifacts_dir

    ablation_results = run_gaussian_ablation(GaussianConfig(seed=args.seed))
    save_ablation_figure(ablation_results, figures_dir / "fig_gaussian_ablation.pdf")
    export_rows_csv(
        build_ablation_rows(ablation_results),
        artifacts_dir / "gaussian_ablation_summary.csv",
    )

    ucurve_result = run_ucurve_experiment()
    save_ucurve_figure(ucurve_result, figures_dir / "fig_gaussian_ucurve.pdf")
    export_rows_csv(
        build_ucurve_rows(ucurve_result), artifacts_dir / "gaussian_ucurve.csv"
    )

    sample_complexity_result = run_sample_complexity_experiment()
    save_sigma_p_complexity_figure(
        sample_complexity_result, figures_dir / "fig_gaussian_sinkhorn.pdf"
    )
    export_rows_csv(
        build_sample_complexity_rows(sample_complexity_result),
        artifacts_dir / "gaussian_sinkhorn.csv",
    )

    runtime_result = run_sinkhorn_runtime_experiment()
    save_sinkhorn_runtime_figure(
        runtime_result, figures_dir / "fig_gaussian_sinkhorn_runtime.pdf"
    )
    export_rows_csv(
        build_sinkhorn_runtime_rows(runtime_result),
        artifacts_dir / "gaussian_sinkhorn_runtime.csv",
    )

    print(f"Saved {figures_dir / 'fig_gaussian_ablation.pdf'}")
    print(f"Saved {figures_dir / 'fig_gaussian_ucurve.pdf'}")
    print(f"Saved {figures_dir / 'fig_gaussian_sinkhorn.pdf'}")
    print(f"Saved {figures_dir / 'fig_gaussian_sinkhorn_runtime.pdf'}")
    print(f"Saved {artifacts_dir / 'gaussian_ablation_summary.csv'}")
    print(f"Saved {artifacts_dir / 'gaussian_ucurve.csv'}")
    print(f"Saved {artifacts_dir / 'gaussian_sinkhorn.csv'}")
    print(f"Saved {artifacts_dir / 'gaussian_sinkhorn_runtime.csv'}")


if __name__ == "__main__":
    main()
