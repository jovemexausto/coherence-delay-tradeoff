from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .common import export_rows_csv
from .glue_theorem_minimal import fixed_span_means


def cusp_kernel_pdf(z: np.ndarray, alpha: float) -> np.ndarray:
    z_abs = np.abs(z)
    norm = (alpha + 1.0) / (2.0 * (alpha + 2.0))
    pdf = norm * (1.0 + z_abs**alpha)
    return np.where(np.abs(z) <= 1.0, pdf, 0.0)


def cusp_kernel_cdf_scalar(z: float, alpha: float) -> float:
    norm = (alpha + 1.0) / (2.0 * (alpha + 2.0))
    if z <= -1.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    if z < 0.0:
        x = -z
        return 0.5 - norm * (x + x ** (alpha + 1.0) / (alpha + 1.0))
    return 0.5 + norm * (z + z ** (alpha + 1.0) / (alpha + 1.0))


def cusp_kernel_cdf(z: np.ndarray, alpha: float) -> np.ndarray:
    vectorized = np.vectorize(lambda value: cusp_kernel_cdf_scalar(float(value), alpha))
    return np.asarray(vectorized(z), dtype=float)


def cusp_kernel_ppf_scalar(u: float, alpha: float) -> float:
    if u <= 0.0:
        return -1.0
    if u >= 1.0:
        return 1.0
    if u < 0.5:
        root = brentq(
            lambda x: cusp_kernel_cdf_scalar(x, alpha) - u,
            -1.0,
            0.0,
            xtol=1e-12,
        )
        return float(root)
    root = brentq(
        lambda x: cusp_kernel_cdf_scalar(x, alpha) - u,
        0.0,
        1.0,
        xtol=1e-12,
    )
    return float(root)


def cusp_kernel_ppf(u: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(
        [cusp_kernel_ppf_scalar(float(value), alpha) for value in u], dtype=float
    )


def shifted_cusp_cdf(
    x_values: np.ndarray, means: np.ndarray, radius: float, alpha: float
) -> np.ndarray:
    z = (x_values[:, None] - means[None, :]) / radius
    return np.mean(cusp_kernel_cdf(z, alpha), axis=1)


def shifted_cusp_pdf(
    x_values: np.ndarray, means: np.ndarray, radius: float, alpha: float
) -> np.ndarray:
    z = (x_values[:, None] - means[None, :]) / radius
    return np.mean(cusp_kernel_pdf(z, alpha), axis=1) / radius


def shifted_cusp_quantiles(
    u: np.ndarray, means: np.ndarray, radius: float, alpha: float
) -> np.ndarray:
    left = float(np.min(means) - radius)
    right = float(np.max(means) + radius)
    return np.asarray(
        [
            brentq(
                lambda x, target=float(value): (
                    shifted_cusp_cdf(
                        np.asarray([x], dtype=float), means, radius, alpha
                    )[0]
                    - target
                ),
                left,
                right,
                xtol=1e-10,
            )
            for value in u
        ],
        dtype=float,
    )


def empirical_cdf_at(sample: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return np.mean(sample[:, None] <= x_values[None, :], axis=0)


def sample_cusp_kernel(rng: np.random.Generator, size: int, alpha: float) -> np.ndarray:
    u = rng.uniform(0.0, 1.0, size=size)
    return cusp_kernel_ppf(u, alpha)


@dataclass(slots=True)
class BahadurRoughKernelConfig:
    support_radius: float = 1.0
    fixed_span: float = 0.5
    alpha_values: tuple[float, ...] = (1.0, 0.5, 0.25)
    n_values: tuple[int, ...] = (25, 50, 100, 200)
    replications: int = 48
    quantile_grid_size: int = 128
    interior_epsilon: float = 0.1


@dataclass(slots=True)
class BahadurRoughKernelResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def run_bahadur_rough_kernel_research(
    config: BahadurRoughKernelConfig | None = None,
) -> BahadurRoughKernelResult:
    if config is None:
        config = BahadurRoughKernelConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    n_values = np.asarray(config.n_values, dtype=int)
    quantile_grid = (
        np.arange(config.quantile_grid_size, dtype=float) + 0.5
    ) / config.quantile_grid_size
    interior_mask = (quantile_grid >= config.interior_epsilon) & (
        quantile_grid <= 1.0 - config.interior_epsilon
    )
    if not np.any(interior_mask):
        raise ValueError("interior_epsilon removes the entire quantile grid")
    interior_grid = quantile_grid[interior_mask]

    for alpha in config.alpha_values:
        residual_curve: list[float] = []
        empirical_curve: list[float] = []
        taylor_curve: list[float] = []
        sup_curve: list[float] = []
        mse_curve: list[float] = []
        recon_curve: list[float] = []

        for n in n_values:
            means = fixed_span_means(n, config.fixed_span, 1.0)
            q_target = shifted_cusp_quantiles(
                quantile_grid, means, config.support_radius, alpha
            )
            f_target = shifted_cusp_pdf(q_target, means, config.support_radius, alpha)
            safe_f = np.maximum(f_target, 1e-10)

            rep_residuals: list[float] = []
            rep_empirical: list[float] = []
            rep_taylor: list[float] = []
            rep_sup: list[float] = []
            rep_mse: list[float] = []
            rep_recon: list[float] = []

            for rep in range(config.replications):
                tri_rng = np.random.default_rng(
                    700_000 + 100 * n + rep + int(1000 * alpha)
                )
                tri_sample = means + config.support_radius * sample_cusp_kernel(
                    tri_rng, n, alpha
                )

                qhat = np.quantile(tri_sample, quantile_grid)
                Fhat_qtarget = empirical_cdf_at(tri_sample, q_target)
                Fhat_qhat = empirical_cdf_at(tri_sample, qhat)
                Fbar_qhat = shifted_cusp_cdf(qhat, means, config.support_radius, alpha)

                line = (quantile_grid - Fhat_qtarget) / safe_f
                residual = qhat - q_target - line
                empirical = (
                    -((Fhat_qhat - Fhat_qtarget) - (Fbar_qhat - quantile_grid)) / safe_f
                )
                taylor = (
                    -((Fbar_qhat - quantile_grid) - safe_f * (qhat - q_target)) / safe_f
                )
                recon = residual - (empirical + taylor)

                rep_residuals.append(float(np.trapezoid(residual**2, quantile_grid)))
                rep_empirical.append(
                    float(np.trapezoid(empirical[interior_mask] ** 2, interior_grid))
                )
                rep_taylor.append(
                    float(np.trapezoid(taylor[interior_mask] ** 2, interior_grid))
                )
                rep_sup.append(float(np.max(np.abs(residual[interior_mask]))))
                rep_mse.append(
                    float(np.trapezoid((qhat - q_target) ** 2, quantile_grid))
                )
                rep_recon.append(
                    float(np.trapezoid(recon[interior_mask] ** 2, interior_grid))
                )

            residual_mean = float(np.mean(rep_residuals))
            empirical_mean = float(np.mean(rep_empirical))
            taylor_mean = float(np.mean(rep_taylor))
            sup_mean = float(np.mean(rep_sup))
            mse_mean = float(np.mean(rep_mse))
            recon_mean = float(np.mean(rep_recon))

            curve_rows.extend(
                [
                    {
                        "experiment": "bahadur-rough-kernel",
                        "alpha": round(float(alpha), 6),
                        "setting": "triangular",
                        "n": n,
                        "value": round(residual_mean, 8),
                    },
                    {
                        "experiment": "bahadur-rough-kernel",
                        "alpha": round(float(alpha), 6),
                        "setting": "empirical",
                        "n": n,
                        "value": round(empirical_mean, 8),
                    },
                    {
                        "experiment": "bahadur-rough-kernel",
                        "alpha": round(float(alpha), 6),
                        "setting": "taylor",
                        "n": n,
                        "value": round(taylor_mean, 8),
                    },
                ]
            )

            residual_curve.append(residual_mean)
            empirical_curve.append(empirical_mean)
            taylor_curve.append(taylor_mean)
            sup_curve.append(sup_mean)
            mse_curve.append(mse_mean)
            recon_curve.append(recon_mean)

        residual_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(residual_curve, dtype=float)), 1
            )[0]
        )
        empirical_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(empirical_curve, dtype=float)), 1
            )[0]
        )
        taylor_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(taylor_curve, dtype=float)), 1
            )[0]
        )
        sup_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(sup_curve, dtype=float)), 1
            )[0]
        )
        mse_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(mse_curve, dtype=float)), 1
            )[0]
        )
        recon_rate = float(
            -np.polyfit(
                np.log(n_values), np.log(np.asarray(recon_curve, dtype=float)), 1
            )[0]
        )

        summary_rows.append(
            {
                "experiment": "bahadur-rough-kernel-rate",
                "alpha": round(float(alpha), 6),
                "residual_rate": round(residual_rate, 6),
                "empirical_rate": round(empirical_rate, 6),
                "taylor_rate": round(taylor_rate, 6),
                "sup_rate": round(sup_rate, 6),
                "mse_rate": round(mse_rate, 6),
                "reconstruction_error_rate": round(recon_rate, 6),
                "last_emp_over_taylor": round(
                    empirical_curve[-1] / max(taylor_curve[-1], 1e-18), 6
                ),
            }
        )

    return BahadurRoughKernelResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bahadur rough-kernel diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/glue_theorem_bahadur_rough_kernel"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bahadur_rough_kernel_research()
    export_rows_csv(
        result.summary_rows,
        args.csv_dir / "glue_theorem_bahadur_rough_kernel_summary.csv",
    )
    export_rows_csv(
        result.curve_rows, args.csv_dir / "glue_theorem_bahadur_rough_kernel_curves.csv"
    )


if __name__ == "__main__":
    main()
