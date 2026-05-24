from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .operational_region_thresholds import (
    certify_operational_epsilon_band,
    maximal_stable_epsilon_band,
)
from .operational_regime_frontier import map_operational_regime


DEFAULT_PAIRS: tuple[tuple[int, int], ...] = ((8, 1), (8, 2), (12, 1), (12, 2))
DEFAULT_EPSILONS: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05)


@dataclass(slots=True)
class BandwiseFrontierResult:
    row_summary: list[dict[str, str | float]]
    band_summary: list[dict[str, str | float]]


def derive_bandwise_sinkhorn_frontier(
    pairs: tuple[tuple[int, int], ...] = DEFAULT_PAIRS,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    sample_sizes: tuple[int, ...] = (24, 48, 96, 160),
    seed_count: int = 24,
    useful_threshold: float = 0.4,
    gap_threshold: float = 0.15,
) -> BandwiseFrontierResult:
    rows = map_operational_regime(
        ambient_intrinsic_pairs=pairs,
        epsilons=epsilons,
        sample_sizes=sample_sizes,
        seed_count=seed_count,
        useful_threshold=useful_threshold,
        gap_threshold=gap_threshold,
    )
    row_summary = [
        {
            "ambient_dim": row.ambient_dim,
            "intrinsic_dim": row.intrinsic_dim,
            "epsilon": row.epsilon,
            "iid_a": round(row.iid_a, 6),
            "triangular_a": round(row.triangular_a, 6),
            "gap": round(row.gap, 6),
            "useful": row.useful,
        }
        for row in rows
    ]
    band_summary = []
    for ambient_dim, intrinsic_dim in pairs:
        epsilon_max = maximal_stable_epsilon_band(
            rows, ambient_dim=ambient_dim, intrinsic_dim=intrinsic_dim
        )
        certificate = (
            None
            if epsilon_max is None
            else certify_operational_epsilon_band(
                rows,
                ambient_dim=ambient_dim,
                intrinsic_dim=intrinsic_dim,
                epsilon_max=epsilon_max,
            )
        )
        band_summary.append(
            {
                "ambient_dim": ambient_dim,
                "intrinsic_dim": intrinsic_dim,
                "epsilon_max": epsilon_max,
                "rows_in_band": "" if certificate is None else certificate.rows_in_band,
                "min_iid_a": "" if certificate is None else certificate.min_iid_a,
                "min_triangular_a": ""
                if certificate is None
                else certificate.min_triangular_a,
                "max_gap": "" if certificate is None else certificate.max_gap,
                "all_useful": "" if certificate is None else certificate.all_useful,
            }
        )
    return BandwiseFrontierResult(row_summary=row_summary, band_summary=band_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the bandwise fixed-epsilon Sinkhorn frontier."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/bandwise_sinkhorn_frontier"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = derive_bandwise_sinkhorn_frontier()
    export_rows_csv(result.row_summary, args.csv_dir / "rows.csv")
    export_rows_csv(result.band_summary, args.csv_dir / "bands.csv")


if __name__ == "__main__":
    main()
