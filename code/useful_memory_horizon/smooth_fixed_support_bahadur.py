from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from .common import export_rows_csv


def smooth_fixed_support_warp(x: np.ndarray, delta: float) -> np.ndarray:
    return x + delta * x * (1.0 - x)


def smooth_fixed_support_inverse(y: np.ndarray, delta: float) -> np.ndarray:
    if abs(delta) < 1e-12:
        return y.copy()
    disc = np.maximum((1.0 + delta) ** 2 - 4.0 * delta * y, 0.0)
    root_plus = ((1.0 + delta) + np.sqrt(disc)) / (2.0 * delta)
    root_minus = ((1.0 + delta) - np.sqrt(disc)) / (2.0 * delta)
    plus_ok = (root_plus >= -1e-12) & (root_plus <= 1.0 + 1e-12)
    minus_ok = (root_minus >= -1e-12) & (root_minus <= 1.0 + 1e-12)
    chosen = np.where(minus_ok, root_minus, root_plus)
    chosen = np.where(plus_ok & ~minus_ok, root_plus, chosen)
    return np.clip(chosen, 0.0, 1.0)


def smooth_fixed_support_pdf(y: np.ndarray, delta: float) -> np.ndarray:
    x = smooth_fixed_support_inverse(y, delta)
    deriv = 1.0 + delta * (1.0 - 2.0 * x)
    return 1.0 / np.maximum(deriv, 1e-12)


def fixed_span_deformations(n: int, span: float, H: float) -> np.ndarray:
    if n <= 1:
        return np.zeros(1, dtype=float)
    raw = np.linspace(0.0, 1.0, n, dtype=float) ** H
    centered = raw - np.mean(raw)
    width = float(np.max(centered) - np.min(centered))
    if width <= 0.0:
        return np.zeros(n, dtype=float)
    return span * centered / width


@dataclass(slots=True)
class SmoothBahadurConfig:
    n_values: tuple[int, ...] = (50, 100, 200, 400, 800)
    replications: int = 48
    quantile_grid_size: int = 256
    interpolation_grid_size: int = 1024
    interior_epsilon: float = 0.02
    span_values: tuple[float, ...] = (0.20, 0.40)
    H_values: tuple[float, ...] = (0.5, 1.0)


@dataclass(slots=True)
class SmoothBahadurResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def _build_target_quantile_tools(
    deformations: np.ndarray,
    interpolation_grid_size: int,
    interior_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, interp1d, interp1d]:
    x_grid = np.linspace(0.0, 1.0, interpolation_grid_size, dtype=float)
    cdf_grid = np.zeros_like(x_grid)
    pdf_grid = np.zeros_like(x_grid)
    for delta in deformations:
        cdf_grid += smooth_fixed_support_inverse(x_grid, float(delta))
        pdf_grid += smooth_fixed_support_pdf(x_grid, float(delta))
    cdf_grid /= float(len(deformations))
    pdf_grid /= float(len(deformations))
    cdf_grid = np.maximum.accumulate(np.clip(cdf_grid, 0.0, 1.0))
    interior = (cdf_grid >= interior_epsilon) & (cdf_grid <= 1.0 - interior_epsilon)
    if np.count_nonzero(interior) < 2:
        raise ValueError("interior grid collapsed; adjust interior_epsilon")
    ppf = interp1d(
        cdf_grid[interior],
        x_grid[interior],
        bounds_error=False,
        fill_value=(float(x_grid[interior][0]), float(x_grid[interior][-1])),
    )
    pdf_at_x = interp1d(
        x_grid,
        np.maximum(pdf_grid, 1e-12),
        bounds_error=False,
        fill_value=(float(pdf_grid[0]), float(pdf_grid[-1])),
    )
    return x_grid, cdf_grid, pdf_grid, ppf, pdf_at_x


def run_smooth_fixed_support_bahadur(
    config: SmoothBahadurConfig | None = None,
) -> SmoothBahadurResult:
    if config is None:
        config = SmoothBahadurConfig()

    quantile_grid = (
        np.arange(config.quantile_grid_size, dtype=float) + 0.5
    ) / config.quantile_grid_size
    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    for span in config.span_values:
        for H in config.H_values:
            residual_curve: list[float] = []
            scaled_curve: list[float] = []
            for n in config.n_values:
                deformations = fixed_span_deformations(n, span, H)
                _, _, _, ppf, pdf_at_x = _build_target_quantile_tools(
                    deformations,
                    config.interpolation_grid_size,
                    config.interior_epsilon,
                )
                q_target = np.asarray(ppf(quantile_grid), dtype=float)
                f_target = np.maximum(
                    np.asarray(pdf_at_x(q_target), dtype=float), 1e-12
                )

                rep_residuals: list[float] = []
                for rep in range(config.replications):
                    rng = np.random.default_rng(
                        10_000_000 + 1000 * n + 100 * rep + int(100 * H)
                    )
                    base_sample = rng.uniform(0.0, 1.0, size=n)
                    tri_sample = smooth_fixed_support_warp(base_sample, deformations)
                    q_hat = np.quantile(tri_sample, quantile_grid)
                    F_hat = np.mean(tri_sample[:, None] <= q_target[None, :], axis=0)
                    linear_term = (quantile_grid - F_hat) / f_target
                    residual = (q_hat - q_target) - linear_term
                    rep_residuals.append(
                        float(np.trapezoid(residual**2, quantile_grid))
                    )

                mean_residual = float(np.mean(rep_residuals))
                residual_curve.append(mean_residual)
                scaled_curve.append(n * mean_residual)
                curve_rows.append(
                    {
                        "experiment": "smooth-fixed-support-bahadur",
                        "span": round(span, 6),
                        "H": round(H, 6),
                        "n": n,
                        "mean_integrated_residual": round(mean_residual, 10),
                        "scaled_n_residual": round(n * mean_residual, 10),
                    }
                )

            log_n = np.log(np.asarray(config.n_values, dtype=float))
            log_residual = np.log(np.maximum(np.asarray(residual_curve), 1e-18))
            log_scaled = np.log(np.maximum(np.asarray(scaled_curve), 1e-18))
            residual_rate = float(-np.polyfit(log_n, log_residual, 1)[0])
            scaled_rate = float(-np.polyfit(log_n, log_scaled, 1)[0])
            summary_rows.append(
                {
                    "experiment": "smooth-fixed-support-bahadur",
                    "span": round(span, 6),
                    "H": round(H, 6),
                    "residual_rate": round(residual_rate, 6),
                    "scaled_n_residual_rate": round(scaled_rate, 6),
                    "last_mean_integrated_residual": round(residual_curve[-1], 10),
                    "last_scaled_n_residual": round(scaled_curve[-1], 10),
                }
            )

    return SmoothBahadurResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smooth fixed-support Bahadur diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/smooth_fixed_support_bahadur"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smooth_fixed_support_bahadur()
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")
    export_rows_csv(result.curve_rows, args.csv_dir / "curves.csv")


if __name__ == "__main__":
    main()
