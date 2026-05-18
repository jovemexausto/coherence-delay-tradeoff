from __future__ import annotations

import math
from dataclasses import dataclass

from .operational_complexity_inheritance import (
    embedded_fixed_span_complexity_inheritance_ratio,
)
from .operational_dual_inheritance_kernel import (
    critical_dual_smoothness_for_parametric_region,
)
from .operational_inheritance_frontier import (
    embedded_fixed_span_operational_iid_constant,
)
from .operational_model_smoothness import squared_euclidean_parametric_region_holds
from .operational_region_thresholds import certify_operational_epsilon_band
from .operational_regime_frontier import OperationalRegimeRow


def finite_class_noniid_empirical_process_constant(
    class_size: int, envelope_bound: float = 1.0
) -> float:
    if class_size <= 0:
        raise ValueError("class_size must be positive")
    if envelope_bound <= 0.0:
        raise ValueError("envelope_bound must be positive")
    return envelope_bound * math.sqrt(2.0 * math.log(2.0 * class_size))


def finite_class_noniid_empirical_process_bound(
    sample_size: int, class_size: int, envelope_bound: float = 1.0
) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    return finite_class_noniid_empirical_process_constant(
        class_size=class_size,
        envelope_bound=envelope_bound,
    ) / math.sqrt(sample_size)


def triangular_operational_inheritance_bound(
    sample_size: int,
    intrinsic_dim: int,
    epsilon: float,
    span: float,
    inheritance_factor: float = 1.0,
) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if inheritance_factor <= 0.0:
        raise ValueError("inheritance_factor must be positive")
    return (
        inheritance_factor
        * embedded_fixed_span_operational_iid_constant(
            intrinsic_dim=intrinsic_dim,
            epsilon=epsilon,
            span=span,
        )
        / math.sqrt(sample_size)
    )


@dataclass(frozen=True, slots=True)
class OperationalRegionCertificate:
    ambient_dim: int
    intrinsic_dim: int
    useful_fraction: float
    min_iid_a: float
    min_triangular_a: float
    max_gap: float
    stable: bool


@dataclass(frozen=True, slots=True)
class StructuralOperationalRegionCertificate:
    ambient_dim: int
    intrinsic_dim: int
    epsilon_max: float
    rows_in_band: int
    holder_smoothness_alpha: float
    critical_smoothness_alpha: float
    exact_complexity_inheritance: bool
    parametric_region_holds: bool
    empirical_band_holds: bool
    min_iid_a: float
    min_triangular_a: float
    carrier_lower_bound: float
    max_gap: float
    theorem_ready: bool


def certify_operational_region(
    rows: tuple[OperationalRegimeRow, ...],
    ambient_dim: int,
    intrinsic_dim: int,
    min_success_fraction: float = 0.75,
) -> OperationalRegionCertificate:
    if not (0.0 <= min_success_fraction <= 1.0):
        raise ValueError("min_success_fraction must lie in [0, 1]")
    pair_rows = tuple(
        row
        for row in rows
        if row.ambient_dim == ambient_dim and row.intrinsic_dim == intrinsic_dim
    )
    if not pair_rows:
        raise ValueError("no rows found for requested ambient/intrinsic pair")
    useful_count = sum(row.useful for row in pair_rows)
    useful_fraction = useful_count / len(pair_rows)
    min_iid_a = min(row.iid_a for row in pair_rows)
    min_triangular_a = min(row.triangular_a for row in pair_rows)
    max_gap = max(row.gap for row in pair_rows)
    return OperationalRegionCertificate(
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        useful_fraction=useful_fraction,
        min_iid_a=min_iid_a,
        min_triangular_a=min_triangular_a,
        max_gap=max_gap,
        stable=useful_fraction >= min_success_fraction,
    )


def certify_structural_operational_region(
    rows: tuple[OperationalRegimeRow, ...],
    ambient_dim: int,
    intrinsic_dim: int,
    epsilon_max: float,
    holder_smoothness_alpha: float,
    span: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
) -> StructuralOperationalRegionCertificate:
    if holder_smoothness_alpha <= 0.0:
        raise ValueError("holder_smoothness_alpha must be positive")
    band_certificate = certify_operational_epsilon_band(
        rows=rows,
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        epsilon_max=epsilon_max,
    )
    critical_smoothness_alpha = critical_dual_smoothness_for_parametric_region(
        intrinsic_dim
    )
    parametric_region_holds = squared_euclidean_parametric_region_holds(
        intrinsic_dim=intrinsic_dim,
        holder_smoothness_alpha=holder_smoothness_alpha,
    )
    exact_complexity_inheritance = True
    band_rows = tuple(
        row
        for row in rows
        if row.ambient_dim == ambient_dim
        and row.intrinsic_dim == intrinsic_dim
        and row.epsilon <= epsilon_max + 1e-12
    )
    for row in band_rows:
        ratio = embedded_fixed_span_complexity_inheritance_ratio(
            intrinsic_dim=intrinsic_dim,
            span=span,
            epsilon=row.epsilon,
            template_radius=template_radius,
            lipschitz_scale=lipschitz_scale,
        )
        if not math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
            exact_complexity_inheritance = False
            break
    carrier_lower_bound = min(
        band_certificate.min_iid_a,
        band_certificate.min_triangular_a,
    )
    theorem_ready = (
        exact_complexity_inheritance
        and parametric_region_holds
        and band_certificate.all_useful
    )
    return StructuralOperationalRegionCertificate(
        ambient_dim=ambient_dim,
        intrinsic_dim=intrinsic_dim,
        epsilon_max=epsilon_max,
        rows_in_band=band_certificate.rows_in_band,
        holder_smoothness_alpha=holder_smoothness_alpha,
        critical_smoothness_alpha=critical_smoothness_alpha,
        exact_complexity_inheritance=exact_complexity_inheritance,
        parametric_region_holds=parametric_region_holds,
        empirical_band_holds=band_certificate.all_useful,
        min_iid_a=band_certificate.min_iid_a,
        min_triangular_a=band_certificate.min_triangular_a,
        carrier_lower_bound=carrier_lower_bound,
        max_gap=band_certificate.max_gap,
        theorem_ready=theorem_ready,
    )


def maximal_theorem_ready_epsilon_band(
    rows: tuple[OperationalRegimeRow, ...],
    ambient_dim: int,
    intrinsic_dim: int,
    holder_smoothness_alpha: float,
    span: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
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
        certificate = certify_structural_operational_region(
            rows=rows,
            ambient_dim=ambient_dim,
            intrinsic_dim=intrinsic_dim,
            epsilon_max=epsilon_max,
            holder_smoothness_alpha=holder_smoothness_alpha,
            span=span,
            template_radius=template_radius,
            lipschitz_scale=lipschitz_scale,
        )
        if certificate.theorem_ready:
            stable = epsilon_max
    return stable
