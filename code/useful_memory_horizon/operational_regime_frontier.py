from __future__ import annotations

from dataclasses import dataclass

from .carrier_roughness_research import (
    CarrierRoughnessResearchConfig,
    run_carrier_roughness_research,
)


@dataclass(frozen=True, slots=True)
class OperationalRegimeRow:
    ambient_dim: int
    intrinsic_dim: int
    epsilon: float
    iid_a: float
    triangular_a: float
    gap: float
    useful: bool


def map_operational_regime(
    ambient_intrinsic_pairs: tuple[tuple[int, int], ...],
    epsilons: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05),
    sample_sizes: tuple[int, ...] = (24, 48, 96, 160),
    seed_count: int = 8,
    useful_threshold: float = 0.4,
    gap_threshold: float = 0.15,
) -> tuple[OperationalRegimeRow, ...]:
    result = run_carrier_roughness_research(
        CarrierRoughnessResearchConfig(
            raw_dims=(),
            ambient_intrinsic_pairs=(),
            raw_sample_sizes=(),
            raw_seed_count=0,
            triangular_dims=(),
            H_values=(),
            fixed_spans=(),
            span_growth_fractions=(),
            triangular_sample_sizes=(),
            triangular_seed_count=0,
            sinkhorn_epsilons=epsilons,
            sinkhorn_ambient_intrinsic_pairs=ambient_intrinsic_pairs,
            sinkhorn_sample_sizes=sample_sizes,
            sinkhorn_seed_count=seed_count,
        )
    )

    rows: list[OperationalRegimeRow] = []
    summary_rows = [
        row for row in result.summary_rows if row["experiment"] == "sinkhorn-fixed-span"
    ]
    for ambient_dim, intrinsic_dim in ambient_intrinsic_pairs:
        for epsilon in epsilons:
            iid_row = next(
                row
                for row in summary_rows
                if f"ambient d={ambient_dim}, intrinsic k={intrinsic_dim}"
                in str(row["setting"])
                and f"eps={epsilon:.2f}" in str(row["setting"])
                and "iid mixture" in str(row["setting"])
            )
            triangular_row = next(
                row
                for row in summary_rows
                if f"ambient d={ambient_dim}, intrinsic k={intrinsic_dim}"
                in str(row["setting"])
                and f"eps={epsilon:.2f}" in str(row["setting"])
                and "triangular" in str(row["setting"])
            )
            iid_a = float(iid_row["carrier_a"])
            triangular_a = float(triangular_row["carrier_a"])
            gap = abs(triangular_a - iid_a)
            rows.append(
                OperationalRegimeRow(
                    ambient_dim=ambient_dim,
                    intrinsic_dim=intrinsic_dim,
                    epsilon=epsilon,
                    iid_a=iid_a,
                    triangular_a=triangular_a,
                    gap=gap,
                    useful=(
                        iid_a > useful_threshold
                        and triangular_a > useful_threshold
                        and gap < gap_threshold
                    ),
                )
            )
    return tuple(rows)
