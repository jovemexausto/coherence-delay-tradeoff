from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
from .gaussian_witness_frontier import (
    exact_gaussian_asymptotic_constant,
    exact_minimal_profile_asymptotic_constant,
)
from .holder_lower_bound_research import (
    Hölder_asymptotic_constant,
    Hölder_optimal_shape_parameter,
    Hölder_scaling_exponents,
    Hölder_witness_bound,
)


def uniform_window_staleness_constant(H: float, n: int) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")
    total = sum(j ** (2.0 * H) for j in range(n))
    return math.sqrt(total / (n ** (2.0 * H + 1.0)))


def asymptotic_staleness_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return 1.0 / math.sqrt(2.0 * H + 1.0)


def gaussian_witness_power(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return 2.0 * H / (2.0 * H + 1.0)


def gaussian_location_upper_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def gaussian_location_upper_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_location_expected_absolute_error(mean: float, sd: float) -> float:
    if sd < 0.0:
        raise ValueError("sd must be non-negative")
    if sd == 0.0:
        return abs(mean)
    standardized_mean = mean / sd
    return sd * (
        2.0 * gaussian_location_upper_normal_pdf(standardized_mean)
        + standardized_mean
        * (2.0 * gaussian_location_upper_normal_cdf(standardized_mean) - 1.0)
    )


def gaussian_location_uniform_mean_risk(
    sigma: float, zeta: float, H: float, n: int
) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if n <= 0:
        raise ValueError("n must be positive")
    bias = zeta * sum((j**H) for j in range(n)) / n
    sd = sigma / math.sqrt(n)
    return gaussian_location_expected_absolute_error(bias, sd)


def gaussian_location_upper_shape_root(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")

    def residual(x: float) -> float:
        return H * x * (
            2.0 * gaussian_location_upper_normal_cdf(x) - 1.0
        ) - gaussian_location_upper_normal_pdf(x)

    lo = 0.0
    hi = 8.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gaussian_location_upper_shape_parameter(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    root = gaussian_location_upper_shape_root(H)
    return ((H + 1.0) * root) ** (2.0 / (2.0 * H + 1.0))


def gaussian_location_upper_asymptotic_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    root = gaussian_location_upper_shape_root(H)
    scale = ((H + 1.0) * root) ** (-1.0 / (2.0 * H + 1.0))
    return scale * (
        2.0 * gaussian_location_upper_normal_pdf(root)
        + root * (2.0 * gaussian_location_upper_normal_cdf(root) - 1.0)
    )


def dirac_uniform_window_staleness(zeta: float, H: float, n: int) -> float:
    if zeta <= 0.0:
        raise ValueError("zeta must be positive")
    return zeta * uniform_window_staleness_constant(H, n) * (n**H)


def supplement_candidate_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return math.exp(-0.5) * (2.0 * H + 1.0) ** (H / (2.0 * H + 1.0))


def supplement_proof_gap_ratio(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return asymptotic_staleness_constant(H) / supplement_candidate_constant(H)


def supplement_proof_gap_shape(a: float, H: float) -> float:
    if a <= 0.0:
        raise ValueError("a must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    carrier_fraction = a / (a + H)
    roughness_fraction = H / (a + H)
    return (carrier_fraction ** (-carrier_fraction)) + (
        roughness_fraction ** (-roughness_fraction)
    )


def supplement_proof_gap(a: float, H: float) -> float:
    if a <= 0.0:
        raise ValueError("a must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    carrier_fraction = a / (a + H)
    return supplement_proof_gap_shape(a, H) * (
        supplement_proof_gap_ratio(H) ** carrier_fraction
    )


def supplement_gap_threshold() -> float:
    lo = 1e-12
    hi = 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if supplement_proof_gap_ratio(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def supplement_proof_gap_infimum_for_H(H: float) -> float:
    ratio = supplement_proof_gap_ratio(H)
    return 2.0 * min(1.0, ratio)


def supplement_proof_gap_global_infimum() -> float:
    return supplement_proof_gap_infimum_for_H(1.0)


def refined_proof_gap_shape(a: float, H: float) -> float:
    return supplement_proof_gap_shape(a, H)


def refined_proof_gap(a: float, H: float, lower_constant: float) -> float:
    if lower_constant <= 0.0:
        raise ValueError("lower_constant must be positive")
    carrier_fraction = a / (a + H)
    ratio = asymptotic_staleness_constant(H) / lower_constant
    return refined_proof_gap_shape(a, H) * (ratio**carrier_fraction)


def gaussian_ramp_proof_gap_ratio(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return asymptotic_staleness_constant(H) / exact_gaussian_asymptotic_constant(H)


def gaussian_ramp_proof_gap(a: float, H: float) -> float:
    return refined_proof_gap(a, H, exact_gaussian_asymptotic_constant(H))


def gaussian_ramp_proof_gap_infimum_for_H(H: float) -> float:
    return 2.0 * min(1.0, gaussian_ramp_proof_gap_ratio(H))


def gaussian_ramp_proof_gap_global_infimum() -> float:
    return 2.0


def gaussian_ramp_piecewise_threshold_H() -> float:
    return math.pi / 4.0 - 0.5


def gaussian_ramp_piecewise_threshold_power() -> float:
    return 1.0 - 2.0 / math.pi


def gaussian_ramp_proof_gap_ratio_lower_bound(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    return (
        math.sqrt(2.0 * math.pi)
        * (1.0 - power) ** ((power + 1.0) / 2.0)
        * (power + 2.0 / math.pi) ** ((power + 1.0) / 2.0)
        / (power**power)
    )


def gaussian_ramp_piecewise_lower_bound(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    threshold = 1.0 - 2.0 / math.pi
    if power <= threshold:
        factor = 2.0 / math.pi
    else:
        factor = 1.0
    return (
        math.sqrt(2.0 * math.pi)
        * factor ** ((power + 1.0) / 2.0)
        * (1.0 - power) ** ((power + 1.0) / 2.0)
        / (power**power)
    )


def gaussian_ramp_piecewise_left_limit_lower_bound() -> float:
    return 2.0


def gaussian_ramp_piecewise_threshold_lower_bound() -> float:
    return gaussian_ramp_piecewise_lower_bound(gaussian_ramp_piecewise_threshold_H())


def gaussian_ramp_piecewise_right_endpoint_lower_bound() -> float:
    return gaussian_ramp_piecewise_lower_bound(1.0)


def gaussian_ramp_piecewise_lower_bound_log_second_derivative(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    return -(1.0 / (2.0 * (1.0 - power))) - 1.0 / ((1.0 - power) ** 2) - 1.0 / power


def gaussian_minimal_proof_gap_ratio(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return asymptotic_staleness_constant(H) / exact_minimal_profile_asymptotic_constant(
        H
    )


def gaussian_minimal_proof_gap(a: float, H: float) -> float:
    return refined_proof_gap(a, H, exact_minimal_profile_asymptotic_constant(H))


def gaussian_minimal_proof_gap_infimum_for_H(H: float) -> float:
    return 2.0 * min(1.0, gaussian_minimal_proof_gap_ratio(H))


def gaussian_minimal_proof_gap_global_infimum() -> float:
    return 2.0


def komatsu_gaussian_tail_upper(x: float) -> float:
    if x < 0.0:
        raise ValueError("x must be non-negative")
    gaussian_density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return 2.0 * gaussian_density / (x + math.sqrt(x * x + 8.0 / math.pi))


def gaussian_minimal_proof_gap_ratio_lower_bound(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    return (
        math.sqrt(2.0 * math.pi)
        * math.sqrt(1.0 - power)
        * (power + 2.0 / math.pi) ** ((power + 1.0) / 2.0)
        / (2.0 - power) ** (power / 2.0)
    )


def gaussian_minimal_proof_gap_ratio_lower_bound_right_endpoint() -> float:
    return gaussian_minimal_proof_gap_ratio_lower_bound(1.0)


def gaussian_minimal_proof_gap_ratio_lower_bound_log_derivative_upper(
    H: float,
) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    mills_constant = 2.0 / math.pi
    return 0.5 * (
        (3.0 * power + mills_constant - 2.0) / (2.0 - power)
        - 1.0 / (1.0 - power)
        + (power + 1.0) / (power + mills_constant)
    )


def gaussian_minimal_log_derivative_upper_numerator(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    mills_constant = 2.0 / math.pi
    return (
        4.0 * power**3
        + (4.0 * mills_constant - 8.0) * power**2
        + (7.0 * mills_constant - mills_constant * mills_constant - 5.0) * power
        + (mills_constant * mills_constant - 4.0 * mills_constant + 2.0)
    )


def gaussian_minimal_log_derivative_upper_numerator_derivative(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    power = gaussian_witness_power(H)
    mills_constant = 2.0 / math.pi
    return (
        12.0 * power**2
        + 2.0 * (4.0 * mills_constant - 8.0) * power
        + (7.0 * mills_constant - mills_constant * mills_constant - 5.0)
    )


@dataclass(slots=True)
class SharpFamilyAuditConfig:
    H_values: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0)
    sigma_zeta_ratios: tuple[float, ...] = (1_000.0, 10_000.0)
    n_values: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 1024)
    max_multiplier: float = 4.0


@dataclass(slots=True)
class SharpFamilyAuditResult:
    staleness_rows: list[dict[str, float]]
    lower_bound_rows: list[dict[str, float]]


def run_sharp_family_audit(
    config: SharpFamilyAuditConfig | None = None,
) -> SharpFamilyAuditResult:
    cfg = config or SharpFamilyAuditConfig()
    staleness_rows: list[dict[str, float]] = []
    lower_bound_rows: list[dict[str, float]] = []

    for H in cfg.H_values:
        asymptotic = asymptotic_staleness_constant(H)
        for n in cfg.n_values:
            finite_n = uniform_window_staleness_constant(H, n)
            staleness_rows.append(
                {
                    "H": round(H, 6),
                    "n": float(n),
                    "finite_n_constant": finite_n,
                    "asymptotic_constant": asymptotic,
                    "relative_error": abs(finite_n - asymptotic) / asymptotic,
                }
            )

        sigma_power, zeta_power = Hölder_scaling_exponents(H)
        asymptotic_lower = Hölder_asymptotic_constant(H)
        supplement_lower = supplement_candidate_constant(H)

        for ratio in cfg.sigma_zeta_ratios:
            sigma = ratio
            zeta = 1.0
            predicted_h = Hölder_optimal_shape_parameter(H) * ratio ** (
                2.0 / (2.0 * H + 1.0)
            )
            h_max = max(10, int(cfg.max_multiplier * predicted_h) + 20)

            best_h = 1
            best_bound = 0.0
            for h in range(1, h_max + 1):
                value = Hölder_witness_bound(sigma, zeta, H, h)
                if value > best_bound:
                    best_h = h
                    best_bound = value

            normalized = best_bound / (sigma**sigma_power * zeta**zeta_power)
            lower_bound_rows.append(
                {
                    "H": round(H, 6),
                    "sigma_zeta_ratio": ratio,
                    "best_h": float(best_h),
                    "predicted_h": predicted_h,
                    "normalized_best": normalized,
                    "current_asymptotic_constant": asymptotic_lower,
                    "supplement_candidate_constant": supplement_lower,
                    "numeric_over_current_ratio": normalized / asymptotic_lower,
                    "numeric_over_supplement_ratio": normalized / supplement_lower,
                }
            )

    return SharpFamilyAuditResult(
        staleness_rows=staleness_rows,
        lower_bound_rows=lower_bound_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run numerical audits for the sharp-constant family frontier."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sharp_family"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sharp_family_audit()
    export_rows_csv(result.staleness_rows, args.csv_dir / "staleness_constants.csv")
    export_rows_csv(result.lower_bound_rows, args.csv_dir / "lower_bound_audit.csv")


if __name__ == "__main__":
    main()
