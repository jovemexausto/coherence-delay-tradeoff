"""Generate the canonical Gaussian tracker figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking.gaussian_tracker import (
    TGTConfig,
    build_ablation_rows,
    build_sample_complexity_rows,
    build_ucurve_rows,
    export_rows_csv,
    run_sample_complexity_experiment,
    run_tgt_ablation,
    run_ucurve_experiment,
    save_ablation_figure,
    save_sigma_p_complexity_figure,
    save_ucurve_figure,
)
from tracking.elec2 import (
    Elec2Config,
    build_elec2_rows,
    run_elec2_experiments,
    save_dynamic_nstar_figure,
    save_elec2_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../figures"),
        help="Directory where PDF figures will be written",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Directory for CSV summaries",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the ablation run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir
    artifacts_dir = args.artifacts_dir

    ablation_results = run_tgt_ablation(TGTConfig(seed=args.seed))
    save_ablation_figure(ablation_results, figures_dir / "fig_ablation.pdf")
    export_rows_csv(
        build_ablation_rows(ablation_results),
        artifacts_dir / "tgt_ablation_summary.csv",
    )

    ucurve_result = run_ucurve_experiment()
    save_ucurve_figure(ucurve_result, figures_dir / "fig_ucurve.pdf")
    export_rows_csv(build_ucurve_rows(ucurve_result), artifacts_dir / "tgt_ucurve.csv")

    sample_complexity_result = run_sample_complexity_experiment()
    save_sigma_p_complexity_figure(
        sample_complexity_result,
        figures_dir / "fig_sinkhorn.pdf",
    )
    export_rows_csv(
        build_sample_complexity_rows(sample_complexity_result),
        artifacts_dir / "tgt_sinkhorn.csv",
    )

    elec2_result = run_elec2_experiments(Elec2Config())
    save_elec2_figure(elec2_result, figures_dir / "fig_elec2.pdf")
    save_dynamic_nstar_figure(elec2_result, figures_dir / "fig_dynamic_nstar.pdf")
    export_rows_csv(build_elec2_rows(elec2_result), artifacts_dir / "tgt_elec2.csv")

    print(f"Saved {figures_dir / 'fig_ablation.pdf'}")
    print(f"Saved {figures_dir / 'fig_ucurve.pdf'}")
    print(f"Saved {figures_dir / 'fig_sinkhorn.pdf'}")
    print(f"Saved {figures_dir / 'fig_elec2.pdf'}")
    print(f"Saved {figures_dir / 'fig_dynamic_nstar.pdf'}")
    print(f"Saved {artifacts_dir / 'tgt_ablation_summary.csv'}")
    print(f"Saved {artifacts_dir / 'tgt_ucurve.csv'}")
    print(f"Saved {artifacts_dir / 'tgt_sinkhorn.csv'}")
    print(f"Saved {artifacts_dir / 'tgt_elec2.csv'}")


if __name__ == "__main__":
    main()
