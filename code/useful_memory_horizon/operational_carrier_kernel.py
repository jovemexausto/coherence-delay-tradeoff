from __future__ import annotations

import math


def mid_covering_number_upper(
    intrinsic_dim: int,
    epsilon: float,
    lipschitz_scale: float = 1.0,
    support_span: float = 1.0,
) -> float:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if lipschitz_scale <= 0.0 or support_span <= 0.0:
        raise ValueError("lipschitz_scale and support_span must be positive")
    resolution = epsilon / lipschitz_scale
    return (1.0 + 2.0 * support_span / resolution) ** intrinsic_dim


def sinkhorn_mid_iid_constant(
    intrinsic_dim: int,
    epsilon: float,
    lipschitz_scale: float = 1.0,
    support_span: float = 1.0,
) -> float:
    return (1.0 + epsilon) * math.sqrt(
        mid_covering_number_upper(
            intrinsic_dim=intrinsic_dim,
            epsilon=epsilon,
            lipschitz_scale=lipschitz_scale,
            support_span=support_span,
        )
    )


def sinkhorn_mid_iid_benchmark(
    sample_size: int,
    intrinsic_dim: int,
    epsilon: float,
    lipschitz_scale: float = 1.0,
    support_span: float = 1.0,
) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    return sinkhorn_mid_iid_constant(
        intrinsic_dim=intrinsic_dim,
        epsilon=epsilon,
        lipschitz_scale=lipschitz_scale,
        support_span=support_span,
    ) / math.sqrt(sample_size)


def sinkhorn_operational_horizon_exponent(H: float) -> float:
    if H <= 0.0:
        raise ValueError("H must be positive")
    return 1.0 / (H + 0.5)


def sinkhorn_operational_horizon_scale(
    H: float,
    roughness_budget: float,
    intrinsic_dim: int,
    epsilon: float,
    lipschitz_scale: float = 1.0,
    support_span: float = 1.0,
) -> float:
    if roughness_budget <= 0.0:
        raise ValueError("roughness_budget must be positive")
    constant = sinkhorn_mid_iid_constant(
        intrinsic_dim=intrinsic_dim,
        epsilon=epsilon,
        lipschitz_scale=lipschitz_scale,
        support_span=support_span,
    )
    return (constant / roughness_budget) ** sinkhorn_operational_horizon_exponent(H)
