"""Entry point for the tracker experiment suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracking.particle_tracker import (
    ABLATION_CONDITIONS,
    TPTActiveBenchmarkConfig,
    TPTConfig,
    build_tcie_calibration_rows,
    build_active_benchmark_rows,
    build_masking_grid_rows,
    build_masking_summary_rows,
    export_summary_csv,
    format_summary_markdown,
    run_tpt_active_benchmark,
    run_coercive_masking_experiment,
    run_tpt_ablation,
    run_tpt_experiment,
    save_active_benchmark_figure,
    save_tcie_calibration_figure,
    save_masking_grid_figure,
    save_coercive_masking_figure,
    save_tpt_ablation_figure,
    save_tpt_figure,
    summarize_result,
)
from tracking.bikes import (
    BikesConfig,
    build_bikes_rows,
    run_bikes_experiments,
    save_bikes_figure,
)
from tracking.kuairand import (
    KuaiRandConfig,
    build_kuairand_summary_rows,
    run_kuairand_active_benchmark,
    save_kuairand_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(
            "demo",
            "ablation",
            "masking",
            "masking-grid",
            "active-benchmark",
            "tcie-calibration",
            "bikes",
            "kuairand-logged",
        ),
        default="demo",
        help="Experiment type to run",
    )
    parser.add_argument(
        "--condition",
        choices=ABLATION_CONDITIONS,
        default="full",
        help="Condition for the demo experiment",
    )
    parser.add_argument("--steps", type=int, default=300, help="Number of time steps")
    parser.add_argument("--particles", type=int, default=750, help="Particle count")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--drift", type=float, default=0.04, help="Mean drift per step")
    parser.add_argument(
        "--influence",
        type=float,
        default=0.0,
        help="How strongly actions pull the environment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../figures/fig_tpt_demo.pdf"),
        help="Output PDF path relative to experiments/",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Directory for CSV/Markdown summaries",
    )
    parser.add_argument(
        "--influences-grid",
        default="0.1,0.2,0.3,0.5,0.7,0.9",
        help="Comma-separated influence values for masking-grid",
    )
    parser.add_argument(
        "--lambdas-grid",
        default="0.5,1.0,2.0,3.0,4.0",
        help="Comma-separated lambda values for masking-grid",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=8,
        help="Number of sequential seeds for masking-grid aggregation",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=12,
        help="Number of seeds for the active benchmark summary",
    )
    parser.add_argument(
        "--calibration-lambdas",
        default="0.5,1.0,2.0,3.0,4.0",
        help="Comma-separated effort-penalty values for effort-corrected score calibration",
    )
    parser.add_argument(
        "--calibration-thresholds",
        default="0.55,0.65,0.75,0.80,0.85,0.90,0.94,0.96,0.98,0.99",
        help="Comma-separated effort-corrected score thresholds for calibration",
    )
    parser.add_argument(
        "--calibration-runs",
        type=int,
        default=8,
        help="Number of seeds for the effort-corrected score calibration sweep",
    )
    parser.add_argument(
        "--kuairand-data-dir",
        type=Path,
        default=Path("../data/kuairand/KuaiRand-Pure/data"),
        help="Path to the extracted KuaiRand-Pure data directory",
    )
    parser.add_argument(
        "--kuairand-window-size",
        type=int,
        default=20,
        help="Rolling window size for KuaiRand signals",
    )
    parser.add_argument(
        "--kuairand-min-phase-count",
        type=int,
        default=20,
        help="Minimum interactions per phase for KuaiRand users",
    )
    parser.add_argument(
        "--kuairand-max-users",
        type=int,
        default=1000,
        help="Maximum KuaiRand users to sample",
    )
    parser.add_argument(
        "--kuairand-threshold-quantile",
        type=float,
        default=0.20,
        help="Healthy-phase quantile used to calibrate KuaiRand thresholds",
    )
    parser.add_argument(
        "--kuairand-tcie-lambda",
        type=float,
        default=3.0,
        help="Effort penalty lambda for KuaiRand effort-corrected score",
    )
    return parser.parse_args()


def parse_float_grid(spec: str) -> list[float]:
    return [float(chunk.strip()) for chunk in spec.split(",") if chunk.strip()]


def main() -> None:
    args = parse_args()
    config = TPTConfig(
        steps=args.steps,
        particles=args.particles,
        seed=args.seed,
        drift=args.drift,
        influence=args.influence,
        condition=args.condition,
    )
    if args.experiment == "masking-grid":
        influences = parse_float_grid(args.influences_grid)
        lambdas = parse_float_grid(args.lambdas_grid)
        seeds = [args.seed + offset for offset in range(args.num_seeds)]
        raw_rows, summary_rows = build_masking_grid_rows(
            config, influences, lambdas, seeds
        )
        raw_csv_path = args.artifacts_dir / "tpt_masking_grid_raw.csv"
        summary_csv_path = args.artifacts_dir / "tpt_masking_grid_summary.csv"
        summary_md_path = args.artifacts_dir / "tpt_masking_grid_summary.md"
        export_summary_csv(raw_rows, raw_csv_path)
        export_summary_csv(summary_rows, summary_csv_path)
        summary_md_path.parent.mkdir(parents=True, exist_ok=True)
        summary_md_path.write_text(
            format_summary_markdown(summary_rows), encoding="utf-8"
        )
        save_masking_grid_figure(summary_rows, influences, lambdas, args.output)
        print(f"Saved masking-grid figure to {args.output}")
        print(f"Saved masking-grid raw CSV to {raw_csv_path}")
        print(f"Saved masking-grid summary CSV to {summary_csv_path}")
        print(f"Saved masking-grid table to {summary_md_path}")
        return

    if args.experiment == "active-benchmark":
        benchmark_config = TPTActiveBenchmarkConfig(
            steps=max(args.steps, 600),
            particles=args.particles,
            seed=args.seed,
            influence=args.influence if args.influence > 0 else 0.3,
        )
        representative = run_tpt_active_benchmark(benchmark_config)
        save_active_benchmark_figure(representative, args.output)
        rows = build_active_benchmark_rows(
            [
                run_tpt_active_benchmark(
                    TPTActiveBenchmarkConfig(
                        steps=benchmark_config.steps,
                        particles=benchmark_config.particles,
                        seed=args.seed + offset,
                        influence=benchmark_config.influence,
                    )
                )
                for offset in range(args.benchmark_runs)
            ]
        )
        csv_path = args.artifacts_dir / "tpt_active_benchmark_summary.csv"
        md_path = args.artifacts_dir / "tpt_active_benchmark_summary.md"
        export_summary_csv(rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(rows), encoding="utf-8")
        print(f"Saved active benchmark figure to {args.output}")
        print(f"Saved active benchmark CSV to {csv_path}")
        print(f"Saved active benchmark table to {md_path}")
        for row in rows:
            print(
                f"{row['phase']} / {row['detector']}: "
                f"rate={row['detection_rate']}, "
                f"median delay={row['median_delay']}"
            )
        return

    if args.experiment == "tcie-calibration":
        lambdas = parse_float_grid(args.calibration_lambdas)
        thresholds = parse_float_grid(args.calibration_thresholds)
        seeds = [args.seed + offset for offset in range(args.calibration_runs)]
        results = []
        for penalty in lambdas:
            for seed in seeds:
                results.append(
                    run_tpt_active_benchmark(
                        TPTActiveBenchmarkConfig(
                            steps=max(args.steps, 600),
                            particles=args.particles,
                            seed=seed,
                            influence=args.influence if args.influence > 0 else 0.3,
                            effort_penalty_lambda=penalty,
                        )
                    )
                )
        rows = build_tcie_calibration_rows(results, lambdas, thresholds)
        csv_path = args.artifacts_dir / "tpt_tcie_calibration.csv"
        md_path = args.artifacts_dir / "tpt_tcie_calibration.md"
        export_summary_csv(rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(rows), encoding="utf-8")
        save_tcie_calibration_figure(rows, args.output)
        print(f"Saved effort-corrected score calibration figure to {args.output}")
        print(f"Saved effort-corrected score calibration CSV to {csv_path}")
        print(f"Saved effort-corrected score calibration table to {md_path}")
        return

    if args.experiment == "bikes":
        results = run_bikes_experiments(BikesConfig())
        save_bikes_figure(results, args.output)
        bikes_rows = build_bikes_rows(results)
        csv_path = args.artifacts_dir / "bikes_summary.csv"
        md_path = args.artifacts_dir / "bikes_summary.md"
        export_summary_csv(bikes_rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(bikes_rows), encoding="utf-8")
        print(f"Saved Bikes figure to {args.output}")
        print(f"Saved Bikes CSV to {csv_path}")
        print(f"Saved Bikes table to {md_path}")
        return

    if args.experiment == "kuairand-logged":
        kuairand_config = KuaiRandConfig(
            data_dir=args.kuairand_data_dir,
            window_size=args.kuairand_window_size,
            min_phase_count=args.kuairand_min_phase_count,
            max_users=args.kuairand_max_users,
            seed=args.seed,
            threshold_quantile=args.kuairand_threshold_quantile,
            tcie_lambda=args.kuairand_tcie_lambda,
        )
        result = run_kuairand_active_benchmark(kuairand_config)
        save_kuairand_figure(result, args.output)
        csv_path = args.artifacts_dir / "kuairand_active_summary.csv"
        md_path = args.artifacts_dir / "kuairand_active_summary.md"
        export_summary_csv(result.summary_rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            format_summary_markdown(result.summary_rows), encoding="utf-8"
        )
        print(f"Saved KuaiRand logged figure to {args.output}")
        print(f"Saved KuaiRand logged CSV to {csv_path}")
        print(f"Saved KuaiRand logged table to {md_path}")
        for row in result.summary_rows:
            print(
                f"{row['phase']} / {row['detector']}: "
                f"rate={row['rate']}, median delay={row['median_delay']}"
            )
        return

    if args.experiment == "masking":
        active_influence = args.influence if args.influence > 0 else 0.3
        results = run_coercive_masking_experiment(
            config, active_influence=active_influence
        )
        save_coercive_masking_figure(results, args.output)
        rows = build_masking_summary_rows(results)
        csv_path = args.artifacts_dir / "tpt_masking_summary.csv"
        md_path = args.artifacts_dir / "tpt_masking_summary.md"
        export_summary_csv(rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(rows), encoding="utf-8")
        print(f"Saved coercive masking figure to {args.output}")
        print(f"Saved masking CSV to {csv_path}")
        print(f"Saved masking table to {md_path}")
        for regime in ("passive", "coercive"):
            summary = summarize_result(results[regime])
            print(
                f"{regime}: "
                f"tail |error|={summary['mean_abs_error']:.3f}, "
                f"action_gap={summary['mean_action_gap']:.3f}, "
                f"effort={summary['mean_effort']:.3f}, "
                f"sigma_P={summary['mean_sigma_p']:.3f}, "
                f"sigma_PE={summary['mean_sigma_p_eff']:.3f}, "
                f"TCI={summary['mean_tci']:.3f}, "
                f"TCIE={summary['mean_tcie']:.3f}"
            )
        return

    if args.experiment == "ablation":
        results = run_tpt_ablation(config)
        save_tpt_ablation_figure(results, args.output)
        print(f"Saved particle-tracker ablation figure to {args.output}")
        for condition in ABLATION_CONDITIONS:
            summary = summarize_result(results[condition])
            print(
                f"{condition}: "
                f"tail |error|={summary['mean_abs_error']:.3f}, "
                f"action_gap={summary['mean_action_gap']:.3f}, "
                f"effort={summary['mean_effort']:.3f}, "
                f"sigma_P={summary['mean_sigma_p']:.3f}, "
                f"sigma_PE={summary['mean_sigma_p_eff']:.3f}, "
                f"sigma_A={summary['mean_sigma_a']:.3f}, "
                f"sigma_Phi={summary['mean_sigma_phi']:.3f}, "
                f"TCI={summary['mean_tci']:.3f}, "
                f"TCIE={summary['mean_tcie']:.3f}, "
                f"resamples={int(summary['resampling_steps'])}"
            )
        return

    result = run_tpt_experiment(config)
    save_tpt_figure(result, args.output)
    summary = summarize_result(result)
        print(f"Saved particle-tracker figure to {args.output}")
    print(
        "Summary: "
        f"tail |error|={summary['mean_abs_error']:.3f}, "
        f"action_gap={summary['mean_action_gap']:.3f}, "
        f"effort={summary['mean_effort']:.3f}, "
        f"sigma_P={summary['mean_sigma_p']:.3f}, "
        f"sigma_PE={summary['mean_sigma_p_eff']:.3f}, "
        f"sigma_A={summary['mean_sigma_a']:.3f}, "
        f"sigma_Phi={summary['mean_sigma_phi']:.3f}, "
        f"TCI={summary['mean_tci']:.3f}, "
        f"TCIE={summary['mean_tcie']:.3f}, "
        f"resamples={int(summary['resampling_steps'])}"
    )


if __name__ == "__main__":
    main()
