from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv


@dataclass(slots=True)
class HolderLowerBoundResearchConfig:
    H_values: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0)
    sigma_zeta_ratios: tuple[float, ...] = (1_000.0, 3_000.0, 10_000.0)
    max_multiplier: float = 4.0


@dataclass(slots=True)
class HolderLowerBoundResearchResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def sum_powers(h: int, exponent: float) -> float:
    return float(sum(r**exponent for r in range(1, h + 1)))


def holder_witness_bound(sigma: float, zeta: float, H: float, h: int) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    if h <= 0:
        raise ValueError("h must be positive")

    path_energy = sum_powers(h, 2.0 * H)
    beta = min(zeta, sigma / (2.0 * math.sqrt(path_energy)))
    kl = 2.0 * beta * beta * path_energy / (sigma * sigma)
    return beta * (h**H) * (1.0 - math.sqrt(kl / 2.0)) / 2.0


def holder_scaling_exponents(H: float) -> tuple[float, float]:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    denominator = 2.0 * H + 1.0
    return 2.0 * H / denominator, 1.0 / denominator


def holder_critical_window_scale(sigma: float, zeta: float, H: float) -> float:
    if sigma <= 0.0 or zeta <= 0.0:
        raise ValueError("sigma and zeta must be positive")
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return (sigma / zeta) ** (2.0 / (2.0 * H + 1.0))


def holder_optimal_shape_parameter(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    exponent = H + 0.5
    shape_power = 2.0 * H * math.sqrt(2.0 * H + 1.0) / (4.0 * H + 1.0)
    return shape_power ** (1.0 / exponent)


def holder_asymptotic_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    a_star = holder_optimal_shape_parameter(H)
    return 0.5 * (a_star**H) * (2.0 * H + 1.0) / (4.0 * H + 1.0)


def holder_predicted_optimal_h(sigma: float, zeta: float, H: float) -> float:
    return holder_optimal_shape_parameter(H) * holder_critical_window_scale(
        sigma, zeta, H
    )


def run_holder_lower_bound_research(
    config: HolderLowerBoundResearchConfig | None = None,
) -> HolderLowerBoundResearchResult:
    if config is None:
        config = HolderLowerBoundResearchConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    for H in config.H_values:
        sigma_power, zeta_power = holder_scaling_exponents(H)
        asymptotic_constant = holder_asymptotic_constant(H)
        for ratio in config.sigma_zeta_ratios:
            sigma = ratio
            zeta = 1.0
            predicted_h = holder_predicted_optimal_h(sigma, zeta, H)
            h_max = max(5, int(config.max_multiplier * predicted_h) + 20)

            best_h = 1
            best_value = 0.0
            path_energy = 0.0
            for h in range(1, h_max + 1):
                path_energy += h ** (2.0 * H)
                beta = min(zeta, sigma / (2.0 * math.sqrt(path_energy)))
                kl = 2.0 * beta * beta * path_energy / (sigma * sigma)
                value = beta * (h**H) * (1.0 - math.sqrt(kl / 2.0)) / 2.0
                curve_rows.append(
                    {
                        "H": round(H, 4),
                        "sigma_zeta_ratio": round(ratio, 4),
                        "h": h,
                        "bound": round(value, 10),
                        "normalized_bound": round(
                            value / (sigma**sigma_power * zeta**zeta_power), 10
                        ),
                        "predicted_h": round(predicted_h, 6),
                    }
                )
                if value > best_value:
                    best_value = value
                    best_h = h

            normalized_best = best_value / (sigma**sigma_power * zeta**zeta_power)
            summary_rows.append(
                {
                    "H": round(H, 4),
                    "sigma_zeta_ratio": round(ratio, 4),
                    "best_h": best_h,
                    "predicted_h": round(predicted_h, 4),
                    "best_bound": round(best_value, 8),
                    "normalized_best": round(normalized_best, 8),
                    "asymptotic_constant": round(asymptotic_constant, 8),
                    "constant_gap": round(normalized_best - asymptotic_constant, 8),
                    "sigma_power": round(sigma_power, 8),
                    "zeta_power": round(zeta_power, 8),
                }
            )

    return HolderLowerBoundResearchResult(
        summary_rows=summary_rows, curve_rows=curve_rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Holder lower-bound witness sweeps."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/holder_lower_bound_research"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_holder_lower_bound_research()
    export_rows_csv(
        result.summary_rows, args.csv_dir / "holder_lower_bound_summary.csv"
    )
    export_rows_csv(result.curve_rows, args.csv_dir / "holder_lower_bound_curves.csv")


if __name__ == "__main__":
    main()
