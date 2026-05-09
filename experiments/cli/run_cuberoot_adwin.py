"""Synthetic benchmark for the cube-root ADWIN detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..cuberoot_adwin.artifacts import save_benchmark_figure
from ..cuberoot_adwin.model import CubeRootADWINBenchmarkConfig, run_benchmark
from ..cuberoot_adwin.reports import (
    build_event_rows,
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
    constant_config = CubeRootADWINBenchmarkConfig(
        seeds=tuple(range(args.seed_offset, args.seed_offset + 20)),
        drift=0.001,
        fixed_window=100,
        fixed_long_window=500,
        ewma_alpha=0.05,
        adwin_delta=0.002,
        Ck=1.0,
        drift_window=100,
    )
    piecewise_config = CubeRootADWINBenchmarkConfig(
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

    save_benchmark_figure(result, args.figures_dir / "fig_cuberoot_adwin.pdf")
    save_benchmark_figure(
        piecewise_result,
        args.figures_dir / "fig_cuberoot_adwin_piecewise.pdf",
    )
    export_summary_csv(
        build_summary_rows(result), args.artifacts_dir / "cuberoot_adwin_summary.csv"
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
    print(f"Saved {args.figures_dir / 'fig_cuberoot_adwin_piecewise.pdf'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_summary.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_events.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_oracle_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_summary.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_events.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_phases.csv'}")
    print(f"Saved {args.artifacts_dir / 'cuberoot_adwin_piecewise_oracle_phases.csv'}")


if __name__ == "__main__":
    main()
