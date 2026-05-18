from __future__ import annotations

import math

import numpy as np

from .gaussian_witness_frontier import (
    exact_minimal_profile_asymptotic_constant,
    exact_minimal_profile_shape_parameter,
)


def regular_family_metric_carrier_exponent(metric_exponent_alpha: float) -> float:
    if metric_exponent_alpha <= 0.0:
        raise ValueError("metric_exponent_alpha must be positive")
    return 0.5 * metric_exponent_alpha


def regular_family_metric_holder_exponent(
    metric_exponent_alpha: float, H: float
) -> float:
    if metric_exponent_alpha <= 0.0:
        raise ValueError("metric_exponent_alpha must be positive")
    if H <= 0.0:
        raise ValueError("H must be positive")
    return metric_exponent_alpha * H


def regular_family_parameter_to_metric_roughness(
    metric_exponent_alpha: float, parameter_roughness: float
) -> float:
    if metric_exponent_alpha <= 0.0:
        raise ValueError("metric_exponent_alpha must be positive")
    if parameter_roughness <= 0.0:
        raise ValueError("parameter_roughness must be positive")
    return parameter_roughness**metric_exponent_alpha


def regular_family_horizon_exponent(metric_exponent_alpha: float, H: float) -> float:
    carrier_exponent = regular_family_metric_carrier_exponent(metric_exponent_alpha)
    metric_holder_exponent = regular_family_metric_holder_exponent(
        metric_exponent_alpha, H
    )
    return 1.0 / (carrier_exponent + metric_holder_exponent)


def regular_family_rate_exponent(metric_exponent_alpha: float, H: float) -> float:
    carrier_exponent = regular_family_metric_carrier_exponent(metric_exponent_alpha)
    metric_holder_exponent = regular_family_metric_holder_exponent(
        metric_exponent_alpha, H
    )
    return carrier_exponent / (carrier_exponent + metric_holder_exponent)


def regular_family_local_metric_scale(
    fisher_information: float, metric_derivative: float = 1.0
) -> float:
    if fisher_information <= 0.0:
        raise ValueError("fisher_information must be positive")
    if metric_derivative <= 0.0:
        raise ValueError("metric_derivative must be positive")
    return metric_derivative / math.sqrt(fisher_information)


def regular_family_parametric_first_moment_constant(
    fisher_information: float, metric_derivative: float = 1.0
) -> float:
    return math.sqrt(2.0 / math.pi) * regular_family_local_metric_scale(
        fisher_information=fisher_information,
        metric_derivative=metric_derivative,
    )


def regular_family_minimal_lower_asymptotic_constant(
    H: float,
    fisher_information: float,
    metric_derivative: float = 1.0,
) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    local_scale = regular_family_local_metric_scale(
        fisher_information=fisher_information,
        metric_derivative=metric_derivative,
    )
    return exact_minimal_profile_asymptotic_constant(H) * local_scale ** (
        2.0 * H / (2.0 * H + 1.0)
    )


def regular_family_minimal_lower_shape_parameter(
    H: float,
    fisher_information: float,
    metric_derivative: float = 1.0,
) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    local_scale = regular_family_local_metric_scale(
        fisher_information=fisher_information,
        metric_derivative=metric_derivative,
    )
    return exact_minimal_profile_shape_parameter(H) * local_scale ** (
        2.0 / (2.0 * H + 1.0)
    )


def gaussian_scale_w2(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    return abs(scale_0 - scale_1)


def gaussian_scale_path_staleness_upper(
    current_scale: float, past_scales: tuple[float, ...]
) -> float:
    if current_scale <= 0.0:
        raise ValueError("current_scale must be positive")
    if not past_scales:
        raise ValueError("past_scales must be non-empty")
    if any(scale <= 0.0 for scale in past_scales):
        raise ValueError("all scales must be positive")
    return math.sqrt(
        sum((scale - current_scale) ** 2 for scale in past_scales) / len(past_scales)
    )


def gaussian_scale_mle(sample: np.ndarray) -> float:
    if sample.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if sample.size == 0:
        raise ValueError("sample must be non-empty")
    return float(np.sqrt(np.mean(sample * sample)))


def gaussian_scale_fisher_information(scale: float) -> float:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return 2.0 / (scale * scale)


def gaussian_scale_mle_asymptotic_constant() -> float:
    return regular_family_parametric_first_moment_constant(
        gaussian_scale_fisher_information(1.0)
    )


def gaussian_scale_minimal_lower_asymptotic_constant(H: float, scale: float) -> float:
    return regular_family_minimal_lower_asymptotic_constant(
        H=H,
        fisher_information=gaussian_scale_fisher_information(scale),
    )


def gaussian_scale_minimal_lower_shape_parameter(H: float, scale: float) -> float:
    return regular_family_minimal_lower_shape_parameter(
        H=H,
        fisher_information=gaussian_scale_fisher_information(scale),
    )


def gaussian_scale_mle_asymptotic_shape_parameter(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return (1.0 / (2.0 * H)) ** (2.0 / (2.0 * H + 1.0))


def gaussian_scale_upper_bound(sigma: float, zeta: float, H: float, n: int) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")
    carrier = gaussian_scale_mle_asymptotic_constant() * sigma / math.sqrt(n)
    staleness = zeta * math.sqrt(sum((j ** (2.0 * H)) for j in range(n)) / n)
    return carrier + staleness


def gaussian_scale_profile(
    current_scale: float, zeta: float, H: float, horizon: int
) -> tuple[float, ...]:
    if current_scale <= 0.0 or zeta <= 0.0:
        raise ValueError("current_scale and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return tuple(current_scale + zeta * (j**H) for j in range(horizon))


def simulate_gaussian_scale_tracking_risk(
    sigma: float,
    zeta: float,
    H: float,
    n: int,
    replications: int,
    seed: int = 0,
) -> float:
    if replications <= 0:
        raise ValueError("replications must be positive")
    scales = gaussian_scale_profile(sigma, zeta, H, n)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replications):
        sample = np.asarray(
            [rng.normal(0.0, scale) for scale in scales],
            dtype=float,
        )
        values.append(abs(gaussian_scale_mle(sample) - sigma))
    return float(np.mean(values))


def estimate_log_slope(sample_sizes: np.ndarray, values: tuple[float, ...]) -> float:
    return float(np.polyfit(np.log(sample_sizes), np.log(np.asarray(values)), 1)[0])
