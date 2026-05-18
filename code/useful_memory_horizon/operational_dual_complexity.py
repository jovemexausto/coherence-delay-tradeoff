from __future__ import annotations

import math


def lca_entropy_exponent(intrinsic_dim: int, smoothness_alpha: float) -> float:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if smoothness_alpha <= 0.0:
        raise ValueError("smoothness_alpha must be positive")
    return intrinsic_dim / smoothness_alpha


def lca_epsilon_prefactor_exponent(
    intrinsic_dim: int, smoothness_alpha: float
) -> float:
    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    if smoothness_alpha <= 0.0:
        raise ValueError("smoothness_alpha must be positive")
    return 0.5 * intrinsic_dim * (smoothness_alpha - 1.0) / smoothness_alpha


def lca_dual_entropy_prefactor(
    intrinsic_dim: int,
    smoothness_alpha: float,
    epsilon: float,
    complexity_constant: float = 1.0,
) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if complexity_constant <= 0.0:
        raise ValueError("complexity_constant must be positive")
    return complexity_constant * (
        min(epsilon, 1.0)
        ** (-lca_epsilon_prefactor_exponent(intrinsic_dim, smoothness_alpha))
    )


def lca_dual_entropy_bound(
    delta: float,
    intrinsic_dim: int,
    smoothness_alpha: float,
    epsilon: float,
    complexity_constant: float = 1.0,
) -> float:
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    return lca_dual_entropy_prefactor(
        intrinsic_dim=intrinsic_dim,
        smoothness_alpha=smoothness_alpha,
        epsilon=epsilon,
        complexity_constant=complexity_constant,
    ) * (delta ** (-lca_entropy_exponent(intrinsic_dim, smoothness_alpha)))


def dudley_entropy_integral_factor(entropy_exponent_beta: float) -> float:
    if entropy_exponent_beta <= 0.0:
        raise ValueError("entropy_exponent_beta must be positive")
    if entropy_exponent_beta >= 2.0:
        raise ValueError("entropy integral diverges when beta >= 2")
    return 1.0 / (1.0 - 0.5 * entropy_exponent_beta)


def lca_rate_exponent(intrinsic_dim: int, smoothness_alpha: float) -> float:
    beta = lca_entropy_exponent(intrinsic_dim, smoothness_alpha)
    if beta < 2.0:
        return 0.5
    if math.isclose(beta, 2.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.5
    return smoothness_alpha / intrinsic_dim


def lca_has_log_correction(intrinsic_dim: int, smoothness_alpha: float) -> bool:
    beta = lca_entropy_exponent(intrinsic_dim, smoothness_alpha)
    return math.isclose(beta, 2.0, rel_tol=0.0, abs_tol=1e-12)


def lca_is_parametric_region(intrinsic_dim: int, smoothness_alpha: float) -> bool:
    return lca_entropy_exponent(intrinsic_dim, smoothness_alpha) < 2.0


def lca_statistical_error_bound(
    sample_size: int,
    intrinsic_dim: int,
    smoothness_alpha: float,
    epsilon: float,
    complexity_constant: float = 1.0,
) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if complexity_constant <= 0.0:
        raise ValueError("complexity_constant must be positive")

    prefactor = lca_dual_entropy_prefactor(
        intrinsic_dim=intrinsic_dim,
        smoothness_alpha=smoothness_alpha,
        epsilon=epsilon,
        complexity_constant=complexity_constant,
    )
    exponent = lca_rate_exponent(intrinsic_dim, smoothness_alpha)
    value = prefactor * (sample_size ** (-exponent))
    if lca_has_log_correction(intrinsic_dim, smoothness_alpha):
        value *= math.log(sample_size + 1.0)
    return value


def lca_noniid_dual_empirical_process_bound(
    sample_size: int,
    intrinsic_dim: int,
    smoothness_alpha: float,
    epsilon: float,
    complexity_constant: float = 1.0,
) -> float:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    beta = lca_entropy_exponent(intrinsic_dim, smoothness_alpha)
    prefactor = lca_dual_entropy_prefactor(
        intrinsic_dim=intrinsic_dim,
        smoothness_alpha=smoothness_alpha,
        epsilon=epsilon,
        complexity_constant=complexity_constant,
    )
    if beta < 2.0:
        return prefactor * dudley_entropy_integral_factor(beta) / math.sqrt(sample_size)
    if math.isclose(beta, 2.0, rel_tol=0.0, abs_tol=1e-12):
        return prefactor * math.log(sample_size + 1.0) / math.sqrt(sample_size)
    return prefactor * (sample_size ** (-smoothness_alpha / intrinsic_dim))


def lca_triangular_inheritance_bound(
    sample_size: int,
    intrinsic_dim: int,
    smoothness_alpha: float,
    epsilon: float,
    inheritance_factor: float = 1.0,
    complexity_constant: float = 1.0,
) -> float:
    if inheritance_factor <= 0.0:
        raise ValueError("inheritance_factor must be positive")
    return inheritance_factor * lca_noniid_dual_empirical_process_bound(
        sample_size=sample_size,
        intrinsic_dim=intrinsic_dim,
        smoothness_alpha=smoothness_alpha,
        epsilon=epsilon,
        complexity_constant=complexity_constant,
    )


def lca_operational_horizon_exponent(
    intrinsic_dim: int, smoothness_alpha: float, H: float
) -> float:
    if H <= 0.0:
        raise ValueError("H must be positive")
    return 1.0 / (lca_rate_exponent(intrinsic_dim, smoothness_alpha) + H)
