from __future__ import annotations

import math

from .operational_dual_complexity import lca_noniid_dual_empirical_process_bound
from .operational_inheritance_frontier import embedded_fixed_span_support_side_lengths


def critical_dual_smoothness_for_parametric_region(intrinsic_dim: int) -> float:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    return 0.5 * intrinsic_dim


def embedded_fixed_span_chart_complexity(
    intrinsic_dim: int,
    span: float,
    holder_smoothness_alpha: float,
    template_radius: float = 1.0,
) -> float:
    if holder_smoothness_alpha <= 0.0:
        raise ValueError("holder_smoothness_alpha must be positive")
    side_lengths = embedded_fixed_span_support_side_lengths(
        intrinsic_dim=intrinsic_dim,
        span=span,
        template_radius=template_radius,
    )
    return sum(side_lengths) ** (intrinsic_dim / holder_smoothness_alpha)


def embedded_fixed_span_dual_noniid_bound(
    sample_size: int,
    intrinsic_dim: int,
    holder_smoothness_alpha: float,
    epsilon: float,
    span: float,
    template_radius: float = 1.0,
) -> float:
    complexity_constant = embedded_fixed_span_chart_complexity(
        intrinsic_dim=intrinsic_dim,
        span=span,
        holder_smoothness_alpha=holder_smoothness_alpha,
        template_radius=template_radius,
    )
    return lca_noniid_dual_empirical_process_bound(
        sample_size=sample_size,
        intrinsic_dim=intrinsic_dim,
        smoothness_alpha=holder_smoothness_alpha,
        epsilon=epsilon,
        complexity_constant=complexity_constant,
    )


def embedded_fixed_span_is_parametric_region(
    intrinsic_dim: int,
    holder_smoothness_alpha: float,
) -> bool:
    return holder_smoothness_alpha > critical_dual_smoothness_for_parametric_region(
        intrinsic_dim
    )
