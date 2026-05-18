from __future__ import annotations

from dataclasses import dataclass

from .operational_regime_frontier import OperationalRegimeRow


@dataclass(frozen=True, slots=True)
class OperationalEpsilonBandCertificate:
    ambient_dim: int
    intrinsic_dim: int
    epsilon_max: float
    rows_in_band: int
    all_useful: bool
    min_iid_a: float
    min_triangular_a: float
    max_gap: float


def certify_operational_epsilon_band(
    rows: tuple[OperationalRegimeRow, ...],
    ambient_dim: int,
    intrinsic_dim: int,
    epsilon_max: float,
) -> OperationalEpsilonBandCertificate:
    band_rows = tuple(
        row
        for row in rows
        if row.ambient_dim == ambient_dim
        and row.intrinsic_dim == intrinsic_dim
        and row.epsilon <= epsilon_max + 1e-12
    )
    if not band_rows:
        raise ValueError("no rows found in requested epsilon band")
    return OperationalEpsilonBandCertificate(
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        epsilon_max=epsilon_max,
        rows_in_band=len(band_rows),
        all_useful=all(row.useful for row in band_rows),
        min_iid_a=min(row.iid_a for row in band_rows),
        min_triangular_a=min(row.triangular_a for row in band_rows),
        max_gap=max(row.gap for row in band_rows),
    )


def maximal_stable_epsilon_band(
    rows: tuple[OperationalRegimeRow, ...],
    ambient_dim: int,
    intrinsic_dim: int,
) -> float | None:
    epsilons = sorted(
        {
            row.epsilon
            for row in rows
            if row.ambient_dim == ambient_dim and row.intrinsic_dim == intrinsic_dim
        }
    )
    if not epsilons:
        raise ValueError("no rows found for requested ambient/intrinsic pair")
    stable = None
    for epsilon_max in epsilons:
        cert = certify_operational_epsilon_band(
            rows,
            ambient_dim=ambient_dim,
            intrinsic_dim=intrinsic_dim,
            epsilon_max=epsilon_max,
        )
        if cert.all_useful:
            stable = epsilon_max
    return stable
