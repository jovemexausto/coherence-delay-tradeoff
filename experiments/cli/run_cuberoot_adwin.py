"""Synthetic benchmark suite for useful memory under drift."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..cuberoot_adwin.artifacts import save_benchmark_figure
from ..cuberoot_adwin.artifacts import save_delay_figure
from ..cuberoot_adwin.artifacts import save_frontier_figure
from ..cuberoot_adwin.artifacts import save_drift_ema_ablation_figure
from ..cuberoot_adwin.artifacts import save_horizon_gap_figure
from ..cuberoot_adwin.artifacts import save_horizon_instability_figure
from ..cuberoot_adwin.model import UMRBenchmarkConfig, run_benchmark
from ..cuberoot_adwin.reports import (
    build_event_rows,
    build_frontier_rows,
    build_delay_rows,
    build_drift_ema_ablation_rows,
    build_horizon_gap_curve_rows,
    build_horizon_instability_rows,
    build_horizon_transition_rows,
    build_oracle_phase_rows,
    build_phase_rows,
    build_summary_rows,
)
from ..core.common import export_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../figures/cuberoot_adwin"),
        help="Directory where PDF figures will be written",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts/cuberoot_adwin"),
        help="Directory for CSV summaries",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constant_config = UMRBenchmarkConfig(
        seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
        drift=0.001,
        fixed_window=100,
        fixed_long_window=500,
        ewma_alpha=0.05,
        adwin_delta=0.002,
        Ck=1.0,
        drift_window=100,
    )
    piecewise_config = UMRBenchmarkConfig(
        seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
        piecewise_drifts=(0.0005, 0.003, 0.001),
        fixed_window=100,
        fixed_long_window=500,
        ewma_alpha=0.05,
        adwin_delta=0.002,
        Ck=1.0,
        drift_window=100,
    )
    result = run_benchmark(constant_config)
    piecewise_result = run_benchmark(piecewise_config)
    alternating_config = UMRBenchmarkConfig(
        seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
        piecewise_drifts=(0.0002, 0.0045, 0.00015, 0.007, 0.00025),
        piecewise_lengths=(900, 140, 700, 70, 1190),
        fixed_window=100,
        fixed_long_window=200,
        ewma_alpha=0.05,
        adwin_delta=0.002,
        Ck=1.0,
        drift_window=100,
    )
    alternating_result = run_benchmark(alternating_config)
    drift_ema_alphas = (0.01, 0.02, 0.05, 0.1, 0.2)
    drift_ema_results = [
        run_benchmark(
            UMRBenchmarkConfig(
                piecewise_drifts=alternating_config.piecewise_drifts,
                piecewise_lengths=alternating_config.piecewise_lengths,
                fixed_window=alternating_config.fixed_window,
                fixed_long_window=alternating_config.fixed_long_window,
                ewma_alpha=alternating_config.ewma_alpha,
                adwin_delta=alternating_config.adwin_delta,
                Ck=alternating_config.Ck,
                drift_window=alternating_config.drift_window,
                drift_ema_alpha=alpha,
                seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
            )
        )
        for alpha in drift_ema_alphas
    ]
    drift_grid = (0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005)
    delay_results = [
        run_benchmark(
            UMRBenchmarkConfig(
                drift=drift,
                fixed_window=100,
                fixed_long_window=500,
                ewma_alpha=0.05,
                adwin_delta=0.002,
                Ck=1.0,
                drift_window=100,
                seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
            )
        )
        for drift in drift_grid
    ]

    save_benchmark_figure(result, args.figures_dir / "fig_cuberoot_adwin.pdf")
    save_frontier_figure(result, args.figures_dir / "fig_lag_variance_frontier.pdf")
    save_delay_figure(
        delay_results, args.figures_dir / "fig_cap_vs_detection_delay.pdf"
    )
    save_horizon_instability_figure(
        alternating_result,
        args.figures_dir / "fig_horizon_instability.pdf",
    )
    save_horizon_gap_figure(result, args.figures_dir / "fig_horizon_gap_cost.pdf")
    save_drift_ema_ablation_figure(
        drift_ema_results,
        list(drift_ema_alphas),
        args.figures_dir / "fig_drift_ema_ablation.pdf",
    )
    save_benchmark_figure(
        piecewise_result,
        args.figures_dir / "fig_cuberoot_adwin_piecewise.pdf",
    )
    export_summary_csv(
        build_summary_rows(result), args.artifacts_dir / "cuberoot_adwin_summary.csv"
    )
    export_summary_csv(
        build_frontier_rows(result), args.artifacts_dir / "cuberoot_adwin_frontier.csv"
    )
    export_summary_csv(
        [row for r in delay_results for row in build_delay_rows(r)],
        args.artifacts_dir / "cuberoot_adwin_delay.csv",
    )
    export_summary_csv(
        build_horizon_instability_rows(alternating_result),
        args.artifacts_dir / "cuberoot_adwin_horizon_instability.csv",
    )
    export_summary_csv(
        build_horizon_transition_rows(alternating_result),
        args.artifacts_dir / "cuberoot_adwin_horizon_transition_ablation.csv",
    )
    export_summary_csv(
        build_horizon_gap_curve_rows(result),
        args.artifacts_dir / "cuberoot_adwin_horizon_gap_cost.csv",
    )
    export_summary_csv(
        build_drift_ema_ablation_rows(drift_ema_results, list(drift_ema_alphas)),
        args.artifacts_dir / "cuberoot_adwin_drift_ema_ablation.csv",
    )
    export_summary_csv(
        build_event_rows(result), args.artifacts_dir / "cuberoot_adwin_events.csv"
    )
    export_summary_csv(
        build_phase_rows(result), args.artifacts_dir / "cuberoot_adwin_phases.csv"
    )
    export_summary_csv(
        build_oracle_phase_rows(result),
        args.artifacts_dir / "cuberoot_adwin_oracle_phases.csv",
    )
    export_summary_csv(
        build_summary_rows(piecewise_result),
        args.artifacts_dir / "cuberoot_adwin_piecewise_summary.csv",
    )
    export_summary_csv(
        build_event_rows(piecewise_result),
        args.artifacts_dir / "cuberoot_adwin_piecewise_events.csv",
    )
    export_summary_csv(
        build_phase_rows(piecewise_result),
        args.artifacts_dir / "cuberoot_adwin_piecewise_phases.csv",
    )
    export_summary_csv(
        build_oracle_phase_rows(piecewise_result),
        args.artifacts_dir / "cuberoot_adwin_piecewise_oracle_phases.csv",
    )

    print(f"Saved {args.figures_dir / 'fig_cuberoot_adwin.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_lag_variance_frontier.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_cap_vs_detection_delay.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_horizon_instability.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_horizon_gap_cost.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_drift_ema_ablation.pdf'}")
    print(f"Saved {args.figures_dir / 'fig_cuberoot_adwin_piecewise.pdf'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_summary.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_frontier.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_delay.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_horizon_instability.csv'}")
    print(
        f"Saved {args.artifacts_dir / 'cuberoot_adwin_horizon_transition_ablation.csv'}"
    )
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_horizon_gap_cost.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_drift_ema_ablation.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_events.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_oracle_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_summary.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_events.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_oracle_phases.csv'}")


if __name__ == "__main__":
    main()
