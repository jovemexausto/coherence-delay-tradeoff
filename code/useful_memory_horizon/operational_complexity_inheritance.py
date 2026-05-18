from __future__ import annotations

from dataclasses import dataclass

from .operational_inheritance_frontier import embedded_fixed_span_support_side_lengths
from .operational_model_smoothness import squared_euclidean_parametric_region_holds
from .operational_regime_frontier import OperationalRegimeRow


def embedded_fixed_span_support_axis_bounds(
    intrinsic_dim: int,
    span: float,
    template_radius: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if span <= 0.0 or template_radius <= 0.0:
        raise ValueError("span and template_radius must be positive")
    first_axis = (-template_radius, span + template_radius)
    if intrinsic_dim == 1:
        return (first_axis,)
    return (first_axis,) + tuple(
        (-template_radius, template_radius) for _ in range(intrinsic_dim - 1)
    )


def triangular_window_support_axis_bounds(
    intrinsic_dim: int,
    span: float,
    template_radius: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    return embedded_fixed_span_support_axis_bounds(
        intrinsic_dim=intrinsic_dim,
        span=span,
        template_radius=template_radius,
    )


def iid_mixture_support_axis_bounds(
    intrinsic_dim: int,
    span: float,
    template_radius: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    return embedded_fixed_span_support_axis_bounds(
        intrinsic_dim=intrinsic_dim,
        span=span,
        template_radius=template_radius,
    )


def axis_aligned_box_covering_upper(
    axis_bounds: tuple[tuple[float, float], ...],
    epsilon: float,
    lipschitz_scale: float = 1.0,
) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if lipschitz_scale <= 0.0:
        raise ValueError("lipschitz_scale must be positive")
    resolution = epsilon / lipschitz_scale
    value = 1.0
    for lower, upper in axis_bounds:
        value *= 1.0 + (upper - lower) / resolution
    return value


def triangular_window_support_covering_upper(
    intrinsic_dim: int,
    span: float,
    epsilon: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
) -> float:
    return axis_aligned_box_covering_upper(
        triangular_window_support_axis_bounds(
            intrinsic_dim=intrinsic_dim,
            span=span,
            template_radius=template_radius,
        ),
        epsilon=epsilon,
        lipschitz_scale=lipschitz_scale,
    )


def iid_mixture_support_covering_upper(
    intrinsic_dim: int,
    span: float,
    epsilon: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
) -> float:
    return axis_aligned_box_covering_upper(
        iid_mixture_support_axis_bounds(
            intrinsic_dim=intrinsic_dim,
            span=span,
            template_radius=template_radius,
        ),
        epsilon=epsilon,
        lipschitz_scale=lipschitz_scale,
    )


def embedded_fixed_span_complexity_inheritance_ratio(
    intrinsic_dim: int,
    span: float,
    epsilon: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
) -> float:
    iid_value = iid_mixture_support_covering_upper(
        intrinsic_dim=intrinsic_dim,
        span=span,
        epsilon=epsilon,
        template_radius=template_radius,
        lipschitz_scale=lipschitz_scale,
    )
    tri_value = triangular_window_support_covering_upper(
        intrinsic_dim=intrinsic_dim,
        span=span,
        epsilon=epsilon,
        template_radius=template_radius,
        lipschitz_scale=lipschitz_scale,
    )
    return tri_value / iid_value


@dataclass(frozen=True, slots=True)
class OperationalTheoremCandidate:
    ambient_dim: int
    intrinsic_dim: int
    epsilon: float
    holder_smoothness_alpha: float
    exact_complexity_inheritance: bool
    parametric_region_holds: bool
    empirical_support: bool
    theorem_ready: bool


def certify_operational_theorem_candidate(
    row: OperationalRegimeRow,
    holder_smoothness_alpha: float,
    span: float,
    template_radius: float = 1.0,
    lipschitz_scale: float = 1.0,
) -> OperationalTheoremCandidate:
    complexity_ratio = embedded_fixed_span_complexity_inheritance_ratio(
        intrinsic_dim=row.intrinsic_dim,
        span=span,
        epsilon=row.epsilon,
        template_radius=template_radius,
        lipschitz_scale=lipschitz_scale,
    )
    exact_complexity_inheritance = abs(complexity_ratio - 1.0) <= 1e-12
    parametric_region_holds = squared_euclidean_parametric_region_holds(
        intrinsic_dim=row.intrinsic_dim,
        holder_smoothness_alpha=holder_smoothness_alpha,
    )
    empirical_support = row.useful
    return OperationalTheoremCandidate(
        ambient_dim=row.ambient_dim,
        intrinsic_dim=row.intrinsic_dim,
        epsilon=row.epsilon,
        holder_smoothness_alpha=holder_smoothness_alpha,
        exact_complexity_inheritance=exact_complexity_inheritance,
        parametric_region_holds=parametric_region_holds,
        empirical_support=empirical_support,
        theorem_ready=(
            exact_complexity_inheritance
            and parametric_region_holds
            and empirical_support
        ),
    )
