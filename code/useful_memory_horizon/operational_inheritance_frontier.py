from __future__ import annotations

import math


def embedded_fixed_span_support_side_lengths(
    intrinsic_dim: int,
    span: float,
    template_radius: float = 1.0,
) -> tuple[float, ...]:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if span <= 0.0 or template_radius <= 0.0:
        raise ValueError("span and template_radius must be positive")
    first_length = span + 2.0 * template_radius
    if intrinsic_dim == 1:
        return (first_length,)
    return (first_length,) + tuple(
        2.0 * template_radius for _ in range(intrinsic_dim - 1)
    )


def embedded_fixed_span_support_covering_upper(
    intrinsic_dim: int,
    epsilon: float,
    span: float,
    lipschitz_scale: float = 1.0,
    template_radius: float = 1.0,
) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if lipschitz_scale <= 0.0:
        raise ValueError("lipschitz_scale must be positive")
    resolution = epsilon / lipschitz_scale
    side_lengths = embedded_fixed_span_support_side_lengths(
        intrinsic_dim=intrinsic_dim,
        span=span,
        template_radius=template_radius,
    )
    value = 1.0
    for length in side_lengths:
        value *= 1.0 + length / resolution
    return value


def embedded_fixed_span_operational_iid_constant(
    intrinsic_dim: int,
    epsilon: float,
    span: float,
    lipschitz_scale: float = 1.0,
    template_radius: float = 1.0,
) -> float:
    return (1.0 + epsilon) * math.sqrt(
        embedded_fixed_span_support_covering_upper(
            intrinsic_dim=intrinsic_dim,
            epsilon=epsilon,
            span=span,
            lipschitz_scale=lipschitz_scale,
            template_radius=template_radius,
        )
    )


def embedded_fixed_span_operational_inheritance_constant(
    intrinsic_dim: int,
    epsilon: float,
    span: float,
    inheritance_factor: float = 1.0,
    lipschitz_scale: float = 1.0,
    template_radius: float = 1.0,
) -> float:
    if inheritance_factor <= 0.0:
        raise ValueError("inheritance_factor must be positive")
    return inheritance_factor * embedded_fixed_span_operational_iid_constant(
        intrinsic_dim=intrinsic_dim,
        epsilon=epsilon,
        span=span,
        lipschitz_scale=lipschitz_scale,
        template_radius=template_radius,
    )


def operational_inheritance_holds(
    intrinsic_dim: int,
    epsilon: float,
    span: float,
    empirical_gap: float,
    gap_threshold: float = 0.15,
) -> bool:
    if empirical_gap < 0.0:
        raise ValueError("empirical_gap must be non-negative")
    return (
        embedded_fixed_span_operational_iid_constant(
            intrinsic_dim=intrinsic_dim,
            epsilon=epsilon,
            span=span,
        )
        > 0.0
        and empirical_gap <= gap_threshold
    )
