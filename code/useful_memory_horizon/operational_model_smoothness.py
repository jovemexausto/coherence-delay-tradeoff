from __future__ import annotations

import math

from .operational_dual_inheritance_kernel import (
    critical_dual_smoothness_for_parametric_region,
)


def embedded_fixed_span_chart_radius(
    span: float, template_radius: float = 1.0
) -> float:
    if span <= 0.0 or template_radius <= 0.0:
        raise ValueError("span and template_radius must be positive")
    return 0.5 * span + template_radius


def squared_euclidean_chart_derivative_bound(
    derivative_order: int,
    span: float,
    template_radius: float = 1.0,
) -> float:
    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")
    radius = embedded_fixed_span_chart_radius(span, template_radius)
    if derivative_order == 0:
        return 4.0 * radius * radius
    if derivative_order == 1:
        return 4.0 * radius
    if derivative_order == 2:
        return 2.0
    return 0.0


def squared_euclidean_supports_holder_smoothness(
    holder_smoothness_alpha: float,
) -> bool:
    if holder_smoothness_alpha <= 0.0:
        raise ValueError("holder_smoothness_alpha must be positive")
    return True


def recommended_parametric_dual_smoothness(
    intrinsic_dim: int, margin: float = 1.0
) -> float:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    return critical_dual_smoothness_for_parametric_region(intrinsic_dim) + margin


def squared_euclidean_parametric_region_holds(
    intrinsic_dim: int,
    holder_smoothness_alpha: float,
) -> bool:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if holder_smoothness_alpha <= 0.0:
        raise ValueError("holder_smoothness_alpha must be positive")
    return squared_euclidean_supports_holder_smoothness(
        holder_smoothness_alpha
    ) and holder_smoothness_alpha > critical_dual_smoothness_for_parametric_region(
        intrinsic_dim
    )
