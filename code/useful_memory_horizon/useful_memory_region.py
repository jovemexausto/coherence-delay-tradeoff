from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True, slots=True)
class UsefulMemoryRegion:
    n_star: float
    lower: float
    upper: float


def horizon_envelope(
    n: float,
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
) -> float:
    if n <= 0.0:
        raise ValueError("n must be positive")
    return C_K * n ** (-a) + C_S * zeta * n**H


def continuous_optimal_horizon(
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
) -> float:
    if C_K <= 0.0 or C_S <= 0.0 or zeta <= 0.0:
        raise ValueError("scale parameters must be positive")
    if a <= 0.0 or H <= 0.0:
        raise ValueError("exponents must be positive")
    return ((a * C_K) / (H * C_S * zeta)) ** (1.0 / (a + H))


def normalized_envelope_ratio(x: float, a: float, H: float) -> float:
    if x <= 0.0:
        raise ValueError("x must be positive")
    if a <= 0.0 or H <= 0.0:
        raise ValueError("exponents must be positive")
    return (H * x ** (-a) + a * x**H) / (a + H)


def _normalized_excess(x: float, a: float, H: float, delta: float) -> float:
    return normalized_envelope_ratio(x, a, H) - (1.0 + delta)


def useful_memory_interval(a: float, H: float, delta: float) -> tuple[float, float]:
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if delta == 0.0:
        return (1.0, 1.0)

    left = brentq(_normalized_excess, 1.0e-12, 1.0, args=(a, H, delta))
    upper = 2.0
    while _normalized_excess(upper, a, H, delta) <= 0.0:
        upper *= 2.0
    right = brentq(_normalized_excess, 1.0, upper, args=(a, H, delta))
    return (left, right)


def useful_memory_bounds(
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
    delta: float,
) -> UsefulMemoryRegion:
    n_star = continuous_optimal_horizon(C_K, a, C_S, zeta, H)
    lower_x, upper_x = useful_memory_interval(a, H, delta)
    return UsefulMemoryRegion(
        n_star=n_star,
        lower=n_star * lower_x,
        upper=n_star * upper_x,
    )
