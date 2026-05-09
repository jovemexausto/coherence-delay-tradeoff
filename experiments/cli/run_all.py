"""Run all experiment domains."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from ..bikes.artifacts import save_bikes_figure
from ..bikes.model import BikesConfig, run_bikes_experiments
from ..bikes.reports import build_bikes_rows
from ..core.harness import ExperimentHarness
from ..core.regime_map import save_regime_first_summary_figure
from ..elec2.artifacts import save_dynamic_nstar_figure, save_elec2_figure
from ..elec2.model import Elec2Config, run_elec2_experiments
from ..elec2.reports import build_elec2_rows
from ..gaussian import (
    GaussianConfig,
    run_gaussian_ablation,
    run_sample_complexity_experiment,
    run_sinkhorn_runtime_experiment,
    run_ucurve_experiment,
)
from ..cuberoot_adwin.artifacts import save_benchmark_figure
from ..cuberoot_adwin.artifacts import save_frontier_figure
from ..cuberoot_adwin.model import CubeRootADWINBenchmarkConfig, run_benchmark
from ..cuberoot_adwin.reports import (
    build_event_rows,
    build_frontier_rows,
    build_oracle_phase_rows,
    build_phase_rows,
    build_summary_rows,
)
from ..gaussian.artifacts import (
    save_ablation_figure,
    save_sigma_p_complexity_figure,
    save_sinkhorn_runtime_figure,
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
    build_sinkhorn_runtime_rows,
    build_ucurve_rows,
)
from ..core.types import SummaryRows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("../figures"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harness = ExperimentHarness(args.figures_dir, args.artifacts_dir)
    harness.ensure()

    particle_config = ParticleConfig(seed=args.seed)
    particle_result = run_particle_experiment(particle_config)
    save_particle_tracking_figure(
        particle_result, harness.figure_path("particle", "fig_particle_demo.pdf")
    )
    print(f"Saved {harness.figure_path('particle', 'fig_particle_demo.pdf')}")

    masking_config = ParticleConfig(seed=args.seed, influence=0.3)
    masking_results = run_particle_coercive_masking_experiment(masking_config)
    save_coercive_masking_figure(
        masking_results, harness.figure_path("particle", "fig_particle_masking.pdf")
    )
    harness.save_summary_csv(
        build_masking_summary_rows(masking_results),
        "particle",
        "particle_masking_summary.csv",
    )

    ablation_results = run_particle_ablation(particle_config)
    save_particle_tracking_ablation_figure(
        ablation_results, harness.figure_path("particle", "fig_particle_ablation.pdf")
    )

    active_benchmark = run_particle_active_benchmark(
        ParticleActiveBenchmarkConfig(seed=args.seed), verbose=True
    )
    save_active_benchmark_figure(
        active_benchmark,
        harness.figure_path("particle", "fig_particle_active_benchmark.pdf"),
    )
    harness.save_summary_csv(
        build_active_benchmark_rows([active_benchmark]),
        "particle",
        "particle_active_benchmark_summary.csv",
    )

    calibration_rows = build_tcie_calibration_rows(
        [active_benchmark], [particle_config.effort_penalty_lambda], [0.8]
    )
    save_tcie_calibration_figure(
        calibration_rows,
        harness.figure_path("particle", "fig_particle_tcie_calibration.pdf"),
    )

    gaussian_results = run_gaussian_ablation(GaussianConfig(seed=args.seed))
    save_ablation_figure(
        gaussian_results, harness.figure_path("gaussian", "fig_gaussian_ablation.pdf")
    )
    harness.save_summary_csv(
        build_ablation_rows(gaussian_results),
        "gaussian",
        "gaussian_ablation_summary.csv",
    )

    ucurve_result = run_ucurve_experiment()
    save_ucurve_figure(
        ucurve_result, harness.figure_path("gaussian", "fig_gaussian_ucurve.pdf")
    )
    harness.save_summary_csv(
        build_ucurve_rows(ucurve_result),
        "gaussian",
        "gaussian_ucurve.csv",
    )

    sample_complexity_result = run_sample_complexity_experiment()
    save_sigma_p_complexity_figure(
        sample_complexity_result,
        harness.figure_path("gaussian", "fig_gaussian_sinkhorn.pdf"),
    )
    harness.save_summary_csv(
        build_sample_complexity_rows(sample_complexity_result),
        "gaussian",
        "gaussian_sinkhorn.csv",
    )

    runtime_result = run_sinkhorn_runtime_experiment()
    save_sinkhorn_runtime_figure(
        runtime_result,
        harness.figure_path("gaussian", "fig_gaussian_sinkhorn_runtime.pdf"),
    )
    harness.save_summary_csv(
        build_sinkhorn_runtime_rows(runtime_result),
        "gaussian",
        "gaussian_sinkhorn_runtime.csv",
    )

    cuberoot_result = run_benchmark(
        CubeRootADWINBenchmarkConfig(
            seeds=tuple(range(args.seed, args.seed + 20)),
            drift=0.001,
            fixed_window=100,
            fixed_long_window=500,
            ewma_alpha=0.05,
            adwin_delta=0.002,
            Ck=1.0,
            drift_window=100,
        )
    )
    save_benchmark_figure(
        cuberoot_result,
        harness.figure_path("cuberoot_adwin", "fig_cuberoot_adwin.pdf"),
    )
    save_frontier_figure(
        cuberoot_result,
        harness.figure_path("cuberoot_adwin", "fig_lag_variance_frontier.pdf"),
    )
    harness.save_summary_csv(
        build_summary_rows(cuberoot_result),
        "cuberoot_adwin",
        "cuberoot_adwin_summary.csv",
    )
    harness.save_summary_csv(
        build_frontier_rows(cuberoot_result),
        "cuberoot_adwin",
        "cuberoot_adwin_frontier.csv",
    )
    harness.save_summary_csv(
        build_event_rows(cuberoot_result),
        "cuberoot_adwin",
        "cuberoot_adwin_events.csv",
    )
    harness.save_summary_csv(
        build_phase_rows(cuberoot_result),
        "cuberoot_adwin",
        "cuberoot_adwin_phases.csv",
    )
    harness.save_summary_csv(
        build_oracle_phase_rows(cuberoot_result),
        "cuberoot_adwin",
        "cuberoot_adwin_oracle_phases.csv",
    )

    bikes_result = run_bikes_experiments(BikesConfig())
    save_bikes_figure(bikes_result, harness.figure_path("bikes", "fig_bikes.pdf"))
    harness.save_summary_csv(
        build_bikes_rows(bikes_result),
        "bikes",
        "bikes_summary.csv",
    )

    elec2_result = run_elec2_experiments(Elec2Config())
    save_elec2_figure(elec2_result, harness.figure_path("elec2", "fig_elec2.pdf"))
    save_dynamic_nstar_figure(
        elec2_result, harness.figure_path("elec2", "fig_dynamic_nstar.pdf")
    )
    harness.save_summary_csv(
        build_elec2_rows(elec2_result),
        "elec2",
        "elec2_summary.csv",
    )

    try:
        kuairand_result = run_kuairand_active_benchmark(KuaiRandConfig())
    except FileNotFoundError:
        print("Skipping KuaiRand: dataset not found.")
        kuairand_rows = cast(
            SummaryRows,
            [
                {
                    "phase": "bubble_detection",
                    "detector": "TCI",
                    "rate": 0.458,
                    "median_delay": 27.0,
                },
                {
                    "phase": "bubble_detection",
                    "detector": "TCIE",
                    "rate": 0.692,
                    "median_delay": 31.8,
                },
                {
                    "phase": "collapse_detection",
                    "detector": "TCI",
                    "rate": 0.548,
                    "median_delay": 37.0,
                },
                {
                    "phase": "collapse_detection",
                    "detector": "TCIE",
                    "rate": 0.700,
                    "median_delay": 45.0,
                },
            ],
        )
    else:
        save_kuairand_figure(
            kuairand_result, harness.figure_path("kuairand", "fig_kuairand.pdf")
        )
        harness.save_summary_csv(
            build_kuairand_summary_rows(kuairand_result.user_results),
            "kuairand",
            "kuairand_summary.csv",
        )
        kuairand_rows = build_kuairand_summary_rows(kuairand_result.user_results)

    save_regime_first_summary_figure(
        build_elec2_rows(elec2_result),
        build_bikes_rows(bikes_result),
        build_active_benchmark_rows([active_benchmark]),
        kuairand_rows,
        harness.figure_file("fig_regime_first_summary.pdf"),
    )
    print(f"Saved {harness.figure_file('fig_regime_first_summary.pdf')}")


if __name__ == "__main__":
    main()
