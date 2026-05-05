"""Particle experiment entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..particle import (
    ParticleActiveBenchmarkConfig,
    ParticleActiveBenchmarkResult,
    ParticleConfig,
    run_particle_ablation,
    run_particle_active_benchmark,
    run_particle_coercive_masking_experiment,
    run_particle_experiment,
)
from ..particle.artifacts import (
    save_active_benchmark_figure,
    save_coercive_masking_figure,
    save_masking_grid_figure,
    save_tcie_calibration_figure,
    save_particle_tracking_ablation_figure,
    save_particle_tracking_figure,
)
from ..particle.reports import (
    build_active_benchmark_rows,
    build_masking_grid_rows,
    build_masking_summary_rows,
    build_tcie_calibration_rows,
    summarize_result,
)
from ..core.common import export_summary_csv, format_summary_markdown


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
        ),
        default="demo",
        help="Experiment type to run",
    )
    parser.add_argument(
        "--condition",
        choices=("full", "fm1", "fm2", "fm3"),
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
        default=Path("../figures/particle/fig_particle_demo.pdf"),
        help="Output PDF path relative to experiments/",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts/particle"),
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
    return parser.parse_args()


def parse_float_grid(spec: str) -> list[float]:
    return [float(chunk.strip()) for chunk in spec.split(",") if chunk.strip()]


def main() -> None:
    args = parse_args()
    config = ParticleConfig(
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
        print(
            f"Running masking-grid: {len(influences)} influences x {len(lambdas)} lambdas x {len(seeds)} seeds",
            flush=True,
        )
        raw_rows, summary_rows = build_masking_grid_rows(
            config, influences, lambdas, seeds
        )
        raw_csv_path = args.artifacts_dir / "particle_masking_grid_raw.csv"
        summary_csv_path = args.artifacts_dir / "particle_masking_grid_summary.csv"
        summary_md_path = args.artifacts_dir / "particle_masking_grid_summary.md"
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
        benchmark_config = ParticleActiveBenchmarkConfig(
            steps=max(args.steps, 600),
            particles=args.particles,
            seed=args.seed,
            influence=args.influence if args.influence > 0 else 0.3,
        )
        print(
            f"Running active benchmark with {args.benchmark_runs} seeds",
            flush=True,
        )
        print("  building representative trace", flush=True)
        representative = run_particle_active_benchmark(benchmark_config, verbose=True)
        print("  representative trace done", flush=True)
        save_active_benchmark_figure(representative, args.output)
        benchmark_results: list[ParticleActiveBenchmarkResult] = []
        for offset in range(args.benchmark_runs):
            print(f"  seed {offset + 1}/{args.benchmark_runs}", flush=True)
            benchmark_results.append(
                run_particle_active_benchmark(
                    ParticleActiveBenchmarkConfig(
                        steps=benchmark_config.steps,
                        particles=benchmark_config.particles,
                        seed=args.seed + offset,
                        influence=benchmark_config.influence,
                    )
                )
            )
        rows = build_active_benchmark_rows(benchmark_results)
        csv_path = args.artifacts_dir / "particle_active_benchmark_summary.csv"
        md_path = args.artifacts_dir / "particle_active_benchmark_summary.md"
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
        results: list[ParticleActiveBenchmarkResult] = []
        print(
            f"Running TCIE calibration: {len(lambdas)} lambdas x {len(seeds)} seeds",
            flush=True,
        )
        for penalty in lambdas:
            print(f"  lambda={penalty}", flush=True)
            for seed in seeds:
                print(f"    seed {seed}", flush=True)
                results.append(
                    run_particle_active_benchmark(
                        ParticleActiveBenchmarkConfig(
                            steps=max(args.steps, 600),
                            particles=args.particles,
                            seed=seed,
                            influence=args.influence if args.influence > 0 else 0.3,
                            effort_penalty_lambda=penalty,
                        )
                    )
                )
        rows = build_tcie_calibration_rows(results, lambdas, thresholds)
        csv_path = args.artifacts_dir / "particle_tcie_calibration.csv"
        md_path = args.artifacts_dir / "particle_tcie_calibration.md"
        export_summary_csv(rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(rows), encoding="utf-8")
        save_tcie_calibration_figure(rows, args.output)
        print(f"Saved calibration figure to {args.output}")
        print(f"Saved calibration CSV to {csv_path}")
        print(f"Saved calibration table to {md_path}")
        return

    if args.experiment == "masking":
        masking_results = run_particle_coercive_masking_experiment(config)
        save_coercive_masking_figure(masking_results, args.output)
        rows = build_masking_summary_rows(masking_results)
        csv_path = args.artifacts_dir / "particle_masking_summary.csv"
        md_path = args.artifacts_dir / "particle_masking_summary.md"
        export_summary_csv(rows, csv_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_summary_markdown(rows), encoding="utf-8")
        print(f"Saved masking figure to {args.output}")
        print(f"Saved masking CSV to {csv_path}")
        print(f"Saved masking table to {md_path}")
        return

    if args.experiment == "ablation":
        ablation_results = run_particle_ablation(config)
        save_particle_tracking_ablation_figure(ablation_results, args.output)
        print(f"Saved ablation figure to {args.output}")
        return

    demo_result = run_particle_experiment(config)
    save_particle_tracking_figure(demo_result, args.output)
    summary = summarize_result(demo_result)
    print(f"Saved figure to {args.output}")
    print(summary)


if __name__ == "__main__":
    main()
