from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .operational_regime_frontier import map_operational_regime
from .operational_theorem_frontier import maximal_theorem_ready_epsilon_band


@dataclass(frozen=True, slots=True)
class TheoremReadyBandRow:
    ambient_dim: int
    intrinsic_dim: int
    holder_smoothness_alpha: float
    span: float
    theorem_ready_epsilon_max: float | None


def run_sinkhorn_theorem_ready_band_report(
    ambient_intrinsic_pairs: tuple[tuple[int, int], ...] = (
        (8, 1),
        (8, 2),
        (12, 1),
        (12, 2),
    ),
    epsilons: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05),
    sample_sizes: tuple[int, ...] = (24, 48, 96, 160),
    seed_count: int = 24,
    holder_smoothness_alpha: float = 2.0,
    span: float = 0.25,
) -> list[dict[str, str | float | None]]:
    rows = map_operational_regime(
        ambient_intrinsic_pairs=ambient_intrinsic_pairs,
        epsilons=epsilons,
        sample_sizes=sample_sizes,
        seed_count=seed_count,
    )
    report_rows: list[dict[str, str | float | None]] = []
    for ambient_dim, intrinsic_dim in ambient_intrinsic_pairs:
        theorem_ready_epsilon_max = maximal_theorem_ready_epsilon_band(
            rows,
            ambient_dim=ambient_dim,
            intrinsic_dim=intrinsic_dim,
            holder_smoothness_alpha=holder_smoothness_alpha,
            span=span,
        )
        report_rows.append(
            {
                "ambient_dim": ambient_dim,
                "intrinsic_dim": intrinsic_dim,
                "holder_smoothness_alpha": holder_smoothness_alpha,
                "span": span,
                "theorem_ready_epsilon_max": theorem_ready_epsilon_max,
            }
        )
    return report_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate theorem-ready Sinkhorn band summaries."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_theorem_ready_band"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sinkhorn_theorem_ready_band_report()
    export_rows_csv(rows, args.csv_dir / "summary.csv")


if __name__ == "__main__":
    main()
