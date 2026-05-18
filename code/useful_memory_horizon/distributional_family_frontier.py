from __future__ import annotations

import math
from dataclasses import dataclass


def gaussian_scale_w2(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    return abs(scale_0 - scale_1)


def gaussian_scale_kl(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    ratio = (scale_0 / scale_1) ** 2
    return 0.5 * (ratio - 1.0 - math.log(ratio))


def gaussian_scale_hellinger_squared(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    return 1.0 - math.sqrt(
        2.0 * scale_0 * scale_1 / (scale_0 * scale_0 + scale_1 * scale_1)
    )


def uniform_scale_w2(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    return abs(scale_0 - scale_1) / math.sqrt(3.0)


def uniform_scale_kl(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    if scale_0 > scale_1:
        return math.inf
    return math.log(scale_1 / scale_0)


def uniform_scale_hellinger_squared(scale_0: float, scale_1: float) -> float:
    if scale_0 <= 0.0 or scale_1 <= 0.0:
        raise ValueError("scales must be positive")
    smaller = min(scale_0, scale_1)
    larger = max(scale_0, scale_1)
    return 1.0 - math.sqrt(smaller / larger)


@dataclass(frozen=True, slots=True)
class LocalGeometryDiagnostic:
    family: str
    delta: float
    w2_over_delta: float
    kl_over_delta_squared: float
    hellinger_over_delta_squared: float


def gaussian_scale_local_diagnostic(
    delta: float, base_scale: float = 1.0
) -> LocalGeometryDiagnostic:
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if base_scale <= 0.0:
        raise ValueError("base_scale must be positive")
    scale_1 = base_scale + delta
    return LocalGeometryDiagnostic(
        family="gaussian_scale",
        delta=delta,
        w2_over_delta=gaussian_scale_w2(base_scale, scale_1) / delta,
        kl_over_delta_squared=gaussian_scale_kl(base_scale, scale_1) / (delta * delta),
        hellinger_over_delta_squared=gaussian_scale_hellinger_squared(
            base_scale, scale_1
        )
        / (delta * delta),
    )


def uniform_scale_local_diagnostic(
    delta: float, base_scale: float = 1.0
) -> LocalGeometryDiagnostic:
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if base_scale <= 0.0:
        raise ValueError("base_scale must be positive")
    scale_1 = base_scale + delta
    return LocalGeometryDiagnostic(
        family="uniform_scale",
        delta=delta,
        w2_over_delta=uniform_scale_w2(base_scale, scale_1) / delta,
        kl_over_delta_squared=uniform_scale_kl(base_scale, scale_1) / (delta * delta),
        hellinger_over_delta_squared=uniform_scale_hellinger_squared(
            base_scale, scale_1
        )
        / (delta * delta),
    )
