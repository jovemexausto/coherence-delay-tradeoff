from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .common import export_rows_csv
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


def dirac_uniform_window_staleness(zeta: float, H: float, n: int) -> float:
    if zeta <= 0.0:
        raise ValueError("zeta must be positive")
    return zeta * uniform_window_staleness_constant(H, n) * (n**H)


def supplement_candidate_constant(H: float) -> float:
    if not (0.0 < H <= 1.0):
        raise ValueError("H must lie in (0, 1]")
    return math.exp(-0.5) * (2.0 * H + 1.0) ** (H / (2.0 * H + 1.0))


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
