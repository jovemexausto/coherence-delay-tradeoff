"""Run all experiment domains."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..bikes.artifacts import save_bikes_figure
from ..bikes.model import BikesConfig, run_bikes_experiments
from ..bikes.reports import build_bikes_rows
from ..core.common import export_summary_csv
from ..elec2.artifacts import save_dynamic_nstar_figure, save_elec2_figure
from ..elec2.model import Elec2Config, run_elec2_experiments
from ..elec2.reports import build_elec2_rows
from ..gaussian import (
    GaussianConfig,
    run_gaussian_ablation,
    run_sample_complexity_experiment,
    run_ucurve_experiment,
)
from ..gaussian.artifacts import (
    save_ablation_figure,
    save_sigma_p_complexity_figure,
    save_ucurve_figure,
)
from ..kuairand.artifacts import save_kuairand_figure
from ..kuairand.model import KuaiRandConfig, run_kuairand_active_benchmark
from ..kuairand.reports import build_kuairand_summary_rows
from ..particle import (
    ParticleActiveBenchmarkConfig,
    ParticleConfig,
    run_particle_ablation,
    run_particle_active_benchmark,
    run_particle_coercive_masking_experiment,
    run_particle_experiment,
)
from ..particle.artifacts import (
    save_active_benchmark_figure,
    save_coercive_masking_figure,
    save_tcie_calibration_figure,
    save_particle_tracking_ablation_figure,
    save_particle_tracking_figure,
)
from ..particle.reports import (
    build_active_benchmark_rows,
    build_masking_summary_rows,
    build_tcie_calibration_rows,
)
from ..gaussian.reports import (
    build_ablation_rows,
    build_sample_complexity_rows,
    build_ucurve_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("../figures"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.figures_dir
    artifacts_dir = args.artifacts_dir

    particle_config = ParticleConfig(seed=args.seed)
    particle_result = run_particle_experiment(particle_config)
    save_particle_tracking_figure(
        particle_result, figures_dir / "particle" / "fig_particle_demo.pdf"
    )
    print(f"Saved {figures_dir / 'particle' / 'fig_particle_demo.pdf'}")

    masking_config = ParticleConfig(seed=args.seed, influence=0.3)
    masking_results = run_particle_coercive_masking_experiment(masking_config)
    save_coercive_masking_figure(
        masking_results, figures_dir / "particle" / "fig_particle_masking.pdf"
    )
    export_summary_csv(
        build_masking_summary_rows(masking_results),
        artifacts_dir / "particle" / "particle_masking_summary.csv",
    )

    ablation_results = run_particle_ablation(particle_config)
    save_particle_tracking_ablation_figure(
        ablation_results, figures_dir / "particle" / "fig_particle_ablation.pdf"
    )

    active_benchmark = run_particle_active_benchmark(
        ParticleActiveBenchmarkConfig(seed=args.seed), verbose=True
    )
    save_active_benchmark_figure(
        active_benchmark, figures_dir / "particle" / "fig_particle_active_benchmark.pdf"
    )
    export_summary_csv(
        build_active_benchmark_rows([active_benchmark]),
        artifacts_dir / "particle" / "particle_active_benchmark_summary.csv",
    )

    calibration_rows = build_tcie_calibration_rows(
        [active_benchmark], [particle_config.effort_penalty_lambda], [0.8]
    )
    save_tcie_calibration_figure(
        calibration_rows, figures_dir / "particle" / "fig_particle_tcie_calibration.pdf"
    )

    gaussian_results = run_gaussian_ablation(GaussianConfig(seed=args.seed))
    save_ablation_figure(
        gaussian_results, figures_dir / "gaussian" / "fig_gaussian_ablation.pdf"
    )
    export_summary_csv(
        build_ablation_rows(gaussian_results),
        artifacts_dir / "gaussian" / "gaussian_ablation_summary.csv",
    )

    ucurve_result = run_ucurve_experiment()
    save_ucurve_figure(
        ucurve_result, figures_dir / "gaussian" / "fig_gaussian_ucurve.pdf"
    )
    export_summary_csv(
        build_ucurve_rows(ucurve_result),
        artifacts_dir / "gaussian" / "gaussian_ucurve.csv",
    )

    sample_complexity_result = run_sample_complexity_experiment()
    save_sigma_p_complexity_figure(
        sample_complexity_result, figures_dir / "gaussian" / "fig_gaussian_sinkhorn.pdf"
    )
    export_summary_csv(
        build_sample_complexity_rows(sample_complexity_result),
        artifacts_dir / "gaussian" / "gaussian_sinkhorn.csv",
    )

    bikes_result = run_bikes_experiments(BikesConfig())
    save_bikes_figure(bikes_result, figures_dir / "bikes" / "fig_bikes.pdf")
    export_summary_csv(
        build_bikes_rows(bikes_result), artifacts_dir / "bikes" / "bikes_summary.csv"
    )

    elec2_result = run_elec2_experiments(Elec2Config())
    save_elec2_figure(elec2_result, figures_dir / "elec2" / "fig_elec2.pdf")
    save_dynamic_nstar_figure(
        elec2_result, figures_dir / "elec2" / "fig_dynamic_nstar.pdf"
    )
    export_summary_csv(
        build_elec2_rows(elec2_result), artifacts_dir / "elec2" / "elec2_summary.csv"
    )

    try:
        kuairand_result = run_kuairand_active_benchmark(KuaiRandConfig())
    except FileNotFoundError:
        print("Skipping KuaiRand: dataset not found.")
    else:
        save_kuairand_figure(
            kuairand_result, figures_dir / "kuairand" / "fig_kuairand.pdf"
        )
        export_summary_csv(
            build_kuairand_summary_rows(kuairand_result.user_results),
            artifacts_dir / "kuairand" / "kuairand_summary.csv",
        )


if __name__ == "__main__":
    main()
