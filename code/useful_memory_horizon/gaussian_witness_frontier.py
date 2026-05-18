from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .holder_lower_bound_research import Hölder_asymptotic_constant


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sum_powers(h: int, exponent: float) -> float:
    if h <= 0:
        raise ValueError("h must be positive")
    return float(sum(r**exponent for r in range(1, h + 1)))


def ramp_profile(H: float, h: int) -> tuple[float, ...]:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if h <= 0:
        raise ValueError("h must be positive")
    return tuple(r**H for r in range(h + 1))


def endpoint_minimal_profile(H: float, h: int) -> tuple[float, ...]:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if h <= 0:
        raise ValueError("h must be positive")
    endpoint = h**H
    return tuple(endpoint - (h - r) ** H for r in range(h + 1))


def present_envelope_minimal_lag_profile(H: float, h: int) -> tuple[float, ...]:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if h <= 0:
        raise ValueError("h must be positive")
    endpoint = h**H
    return tuple(endpoint - (j**H) for j in range(h + 1))


def profile_energy(profile: tuple[float, ...]) -> float:
    if len(profile) < 2:
        raise ValueError("profile must include at least two points")
    return float(sum(value * value for value in profile[1:]))


def ramp_energy_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return 1.0 / (2.0 * H + 1.0)


def minimal_endpoint_energy_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return 2.0 * H * H / ((H + 1.0) * (2.0 * H + 1.0))


def fixed_h_beta_root() -> float:
    """Solve Phi(-x) = x phi(x)."""

    def residual(x: float) -> float:
        return normal_cdf(-x) - x * normal_pdf(x)

    lo = 0.0
    hi = 8.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def asymptotic_shape_root(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")

    power = 2.0 * H / (2.0 * H + 1.0)

    def residual(x: float) -> float:
        return power * normal_cdf(-x) - x * normal_pdf(x)

    lo = 0.0
    hi = 8.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def exact_fixed_h_optimal_beta(sigma: float, zeta: float, H: float, h: int) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    path_energy = sum_powers(h, 2.0 * H)
    unconstrained = fixed_h_beta_root() * sigma / math.sqrt(path_energy)
    return min(zeta, unconstrained)


def exact_gaussian_witness_bound(sigma: float, zeta: float, H: float, h: int) -> float:
    beta = exact_fixed_h_optimal_beta(sigma, zeta, H, h)
    path_energy = sum_powers(h, 2.0 * H)
    testing_error = normal_cdf(-beta * math.sqrt(path_energy) / sigma)
    return beta * (h**H) * testing_error


def exact_profile_witness_bound(
    sigma: float, zeta: float, profile: tuple[float, ...]
) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    endpoint = profile[-1]
    path_energy = profile_energy(profile)
    beta = min(zeta, fixed_h_beta_root() * sigma / math.sqrt(path_energy))
    testing_error = normal_cdf(-beta * math.sqrt(path_energy) / sigma)
    return beta * endpoint * testing_error


def exact_gaussian_shape_parameter(H: float) -> float:
    return exact_profile_shape_parameter(H, ramp_energy_constant(H))


def exact_profile_shape_parameter(H: float, energy_constant: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if energy_constant <= 0.0:
        raise ValueError("energy_constant must be positive")
    x_star = asymptotic_shape_root(H)
    return (x_star / math.sqrt(energy_constant)) ** (2.0 / (2.0 * H + 1.0))


def exact_minimal_profile_shape_parameter(H: float) -> float:
    return exact_profile_shape_parameter(H, minimal_endpoint_energy_constant(H))


def exact_gaussian_asymptotic_constant(H: float) -> float:
    return exact_profile_asymptotic_constant(H, ramp_energy_constant(H))


def exact_profile_asymptotic_constant(H: float, energy_constant: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if energy_constant <= 0.0:
        raise ValueError("energy_constant must be positive")
    x_star = asymptotic_shape_root(H)
    power = 2.0 * H / (2.0 * H + 1.0)
    return (
        (energy_constant ** (-H / (2.0 * H + 1.0)))
        * (x_star**power)
        * normal_cdf(-x_star)
    )


def exact_minimal_profile_asymptotic_constant(H: float) -> float:
    return exact_profile_asymptotic_constant(H, minimal_endpoint_energy_constant(H))


@dataclass(slots=True)
class GaussianWitnessFrontierConfig:
    H_values: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0)
    sigma_zeta_ratios: tuple[float, ...] = (1_000.0, 10_000.0)
    max_multiplier: float = 4.0


@dataclass(slots=True)
class GaussianWitnessFrontierResult:
    summary_rows: list[dict[str, float]]
    curve_rows: list[dict[str, float]]
    comparison_rows: list[dict[str, float | str]]


def _minimal_profile_energy_from_sums(
    h: int, H: float, sum_h: float, sum_2h: float
) -> float:
    endpoint = h**H
    return h * endpoint * endpoint - 2.0 * endpoint * sum_h + sum_2h


def run_gaussian_witness_frontier(
    config: GaussianWitnessFrontierConfig | None = None,
) -> GaussianWitnessFrontierResult:
    cfg = config or GaussianWitnessFrontierConfig()
    summary_rows: list[dict[str, float]] = []
    curve_rows: list[dict[str, float]] = []
    comparison_rows: list[dict[str, float | str]] = []

    for H in cfg.H_values:
        sigma_power = 2.0 * H / (2.0 * H + 1.0)
        zeta_power = 1.0 / (2.0 * H + 1.0)
        asymptotic_constant = exact_gaussian_asymptotic_constant(H)
        shape_parameter = exact_gaussian_shape_parameter(H)
        current_constant = Hölder_asymptotic_constant(H)

        for ratio in cfg.sigma_zeta_ratios:
            sigma = ratio
            zeta = 1.0
            predicted_h = shape_parameter * ratio ** (2.0 / (2.0 * H + 1.0))
            h_max = max(10, int(cfg.max_multiplier * predicted_h) + 20)

            best_h = 1
            best_value = 0.0
            best_beta = zeta
            root = fixed_h_beta_root()
            path_energy = 0.0
            for h in range(1, h_max + 1):
                path_energy += h ** (2.0 * H)
                beta = min(zeta, root * sigma / math.sqrt(path_energy))
                testing_error = normal_cdf(-beta * math.sqrt(path_energy) / sigma)
                value = beta * (h**H) * testing_error
                normalized = value / (sigma**sigma_power * zeta**zeta_power)
                curve_rows.append(
                    {
                        "H": round(H, 6),
                        "sigma_zeta_ratio": ratio,
                        "h": float(h),
                        "beta": beta,
                        "bound": value,
                        "normalized_bound": normalized,
                        "predicted_h": predicted_h,
                    }
                )
                if value > best_value:
                    best_h = h
                    best_value = value
                    best_beta = beta

            normalized_best = best_value / (sigma**sigma_power * zeta**zeta_power)
            summary_rows.append(
                {
                    "H": round(H, 6),
                    "sigma_zeta_ratio": ratio,
                    "best_h": float(best_h),
                    "predicted_h": predicted_h,
                    "best_beta": best_beta,
                    "normalized_best": normalized_best,
                    "exact_asymptotic_constant": asymptotic_constant,
                    "current_pinsker_constant": current_constant,
                    "numeric_over_exact_ratio": normalized_best / asymptotic_constant,
                    "numeric_over_pinsker_ratio": normalized_best / current_constant,
                }
            )

            minimal_constant = exact_minimal_profile_asymptotic_constant(H)
            minimal_shape = exact_minimal_profile_shape_parameter(H)
            predicted_minimal_h = minimal_shape * ratio ** (2.0 / (2.0 * H + 1.0))
            h_max_minimal = max(10, int(cfg.max_multiplier * predicted_minimal_h) + 20)
            best_minimal_h = 1
            best_minimal_value = 0.0
            sum_h = 0.0
            sum_2h = 0.0
            for h in range(1, h_max_minimal + 1):
                if h > 1:
                    sum_h += (h - 1) ** H
                    sum_2h += (h - 1) ** (2.0 * H)
                path_energy = _minimal_profile_energy_from_sums(h, H, sum_h, sum_2h)
                beta = min(zeta, root * sigma / math.sqrt(path_energy))
                testing_error = normal_cdf(-beta * math.sqrt(path_energy) / sigma)
                value = beta * (h**H) * testing_error
                if value > best_minimal_value:
                    best_minimal_h = h
                    best_minimal_value = value

            normalized_minimal = best_minimal_value / (
                sigma**sigma_power * zeta**zeta_power
            )
            comparison_rows.extend(
                [
                    {
                        "profile": "ramp",
                        "H": round(H, 6),
                        "sigma_zeta_ratio": ratio,
                        "best_h": float(best_h),
                        "predicted_h": predicted_h,
                        "normalized_best": normalized_best,
                        "asymptotic_constant": asymptotic_constant,
                        "numeric_over_asymptotic_ratio": normalized_best
                        / asymptotic_constant,
                    },
                    {
                        "profile": "endpoint_minimal",
                        "H": round(H, 6),
                        "sigma_zeta_ratio": ratio,
                        "best_h": float(best_minimal_h),
                        "predicted_h": predicted_minimal_h,
                        "normalized_best": normalized_minimal,
                        "asymptotic_constant": minimal_constant,
                        "numeric_over_asymptotic_ratio": normalized_minimal
                        / minimal_constant,
                    },
                ]
            )

    return GaussianWitnessFrontierResult(
        summary_rows=summary_rows,
        curve_rows=curve_rows,
        comparison_rows=comparison_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact Gaussian witness lower-bound frontier sweeps."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/gaussian_witness_frontier"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_gaussian_witness_frontier()
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")
    export_rows_csv(result.curve_rows, args.csv_dir / "curves.csv")
    export_rows_csv(result.comparison_rows, args.csv_dir / "profile_comparison.csv")


if __name__ == "__main__":
    main()
