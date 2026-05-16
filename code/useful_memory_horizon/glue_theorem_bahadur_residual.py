from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .glue_theorem_minimal import (
    fixed_span_means,
    uniform_mixture_cdf,
    uniform_mixture_quantiles,
)


@dataclass(slots=True)
class BahadurResidualConfig:
    support_radius: float = 1.0
    fixed_span: float = 0.5
    n_values: tuple[int, ...] = (25, 50, 100, 200, 400)
    replications: int = 96
    quantile_grid_size: int = 256
    interior_epsilon: float = 0.1


@dataclass(slots=True)
class BahadurResidualResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def target_pdf(u: np.ndarray, means: np.ndarray, radius: float) -> np.ndarray:
    q = uniform_mixture_quantiles(u, means, radius)
    # Uniform kernels: density is the fraction of cells for which q is inside [mu-r, mu+r]
    inside = np.abs(q[:, None] - means[None, :]) <= radius
    return np.mean(inside, axis=1) / (2.0 * radius)


def empirical_cdf_at(sample: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return np.mean(sample[:, None] <= x_values[None, :], axis=0)


def run_bahadur_residual_research(
    config: BahadurResidualConfig | None = None,
) -> BahadurResidualResult:
    if config is None:
        config = BahadurResidualConfig()

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

    tri_sup_curve: list[float] = []
    iid_sup_curve: list[float] = []
    tri_interior_residual_curve: list[float] = []
    iid_interior_residual_curve: list[float] = []
    tri_interior_sup_curve: list[float] = []
    iid_interior_sup_curve: list[float] = []
    tri_empirical_curve: list[float] = []
    iid_empirical_curve: list[float] = []
    tri_taylor_curve: list[float] = []
    iid_taylor_curve: list[float] = []
    tri_recon_error_curve: list[float] = []
    iid_recon_error_curve: list[float] = []

    for n in n_values:
        means = fixed_span_means(n, config.fixed_span, 1.0)
        q_target = uniform_mixture_quantiles(
            quantile_grid, means, config.support_radius
        )
        f_target = target_pdf(quantile_grid, means, config.support_radius)
        q_target_interior = q_target[interior_mask]
        safe_f = np.maximum(f_target, 1e-10)

        tri_residuals: list[float] = []
        iid_residuals: list[float] = []
        tri_sup_residuals: list[float] = []
        iid_sup_residuals: list[float] = []
        tri_interior_residuals: list[float] = []
        iid_interior_residuals: list[float] = []
        tri_interior_sup_residuals: list[float] = []
        iid_interior_sup_residuals: list[float] = []
        tri_mse: list[float] = []
        iid_mse: list[float] = []
        tri_interior_mse: list[float] = []
        iid_interior_mse: list[float] = []
        tri_linear: list[float] = []
        iid_linear: list[float] = []
        tri_interior_linear: list[float] = []
        iid_interior_linear: list[float] = []
        tri_empirical_terms: list[float] = []
        iid_empirical_terms: list[float] = []
        tri_taylor_terms: list[float] = []
        iid_taylor_terms: list[float] = []
        tri_recon_errors: list[float] = []
        iid_recon_errors: list[float] = []

        for rep in range(config.replications):
            tri_rng = np.random.default_rng(300_000 + 100 * n + rep)
            iid_rng = np.random.default_rng(400_000 + 100 * n + rep)

            tri_sample = means + tri_rng.uniform(
                -config.support_radius, config.support_radius, size=n
            )
            iid_idx = iid_rng.integers(0, n, size=n)
            iid_sample = means[iid_idx] + iid_rng.uniform(
                -config.support_radius, config.support_radius, size=n
            )

            tri_qhat = np.quantile(tri_sample, quantile_grid)
            iid_qhat = np.quantile(iid_sample, quantile_grid)

            tri_Fhat = empirical_cdf_at(tri_sample, q_target)
            iid_Fhat = empirical_cdf_at(iid_sample, q_target)
            tri_Fhat_qhat = empirical_cdf_at(tri_sample, tri_qhat)
            iid_Fhat_qhat = empirical_cdf_at(iid_sample, iid_qhat)
            tri_Fbar_qhat = np.asarray(
                [
                    uniform_mixture_cdf(float(x), means, config.support_radius)
                    for x in tri_qhat
                ],
                dtype=float,
            )
            iid_Fbar_qhat = np.asarray(
                [
                    uniform_mixture_cdf(float(x), means, config.support_radius)
                    for x in iid_qhat
                ],
                dtype=float,
            )

            tri_line = (quantile_grid - tri_Fhat) / safe_f
            iid_line = (quantile_grid - iid_Fhat) / safe_f

            tri_R = tri_qhat - q_target - tri_line
            iid_R = iid_qhat - q_target - iid_line
            tri_empirical = (
                -((tri_Fhat_qhat - tri_Fhat) - (tri_Fbar_qhat - quantile_grid)) / safe_f
            )
            iid_empirical = (
                -((iid_Fhat_qhat - iid_Fhat) - (iid_Fbar_qhat - quantile_grid)) / safe_f
            )
            tri_taylor = (
                -((tri_Fbar_qhat - quantile_grid) - safe_f * (tri_qhat - q_target))
                / safe_f
            )
            iid_taylor = (
                -((iid_Fbar_qhat - quantile_grid) - safe_f * (iid_qhat - q_target))
                / safe_f
            )
            tri_recon = tri_R - (tri_empirical + tri_taylor)
            iid_recon = iid_R - (iid_empirical + iid_taylor)
            tri_R_interior = tri_R[interior_mask]
            iid_R_interior = iid_R[interior_mask]
            tri_empirical_interior = tri_empirical[interior_mask]
            iid_empirical_interior = iid_empirical[interior_mask]
            tri_taylor_interior = tri_taylor[interior_mask]
            iid_taylor_interior = iid_taylor[interior_mask]
            tri_recon_interior = tri_recon[interior_mask]
            iid_recon_interior = iid_recon[interior_mask]

            tri_residuals.append(float(np.trapezoid(tri_R**2, quantile_grid)))
            iid_residuals.append(float(np.trapezoid(iid_R**2, quantile_grid)))
            tri_sup_residuals.append(float(np.max(np.abs(tri_R))))
            iid_sup_residuals.append(float(np.max(np.abs(iid_R))))
            tri_interior_residuals.append(
                float(np.trapezoid(tri_R_interior**2, interior_grid))
            )
            iid_interior_residuals.append(
                float(np.trapezoid(iid_R_interior**2, interior_grid))
            )
            tri_interior_sup_residuals.append(float(np.max(np.abs(tri_R_interior))))
            iid_interior_sup_residuals.append(float(np.max(np.abs(iid_R_interior))))
            tri_mse.append(
                float(np.trapezoid((tri_qhat - q_target) ** 2, quantile_grid))
            )
            iid_mse.append(
                float(np.trapezoid((iid_qhat - q_target) ** 2, quantile_grid))
            )
            tri_interior_mse.append(
                float(
                    np.trapezoid(
                        (tri_qhat[interior_mask] - q_target_interior) ** 2,
                        interior_grid,
                    )
                )
            )
            iid_interior_mse.append(
                float(
                    np.trapezoid(
                        (iid_qhat[interior_mask] - q_target_interior) ** 2,
                        interior_grid,
                    )
                )
            )
            tri_linear.append(float(np.trapezoid(tri_line**2, quantile_grid)))
            iid_linear.append(float(np.trapezoid(iid_line**2, quantile_grid)))
            tri_interior_linear.append(
                float(np.trapezoid((tri_line[interior_mask]) ** 2, interior_grid))
            )
            iid_interior_linear.append(
                float(np.trapezoid((iid_line[interior_mask]) ** 2, interior_grid))
            )
            tri_empirical_terms.append(
                float(np.trapezoid(tri_empirical_interior**2, interior_grid))
            )
            iid_empirical_terms.append(
                float(np.trapezoid(iid_empirical_interior**2, interior_grid))
            )
            tri_taylor_terms.append(
                float(np.trapezoid(tri_taylor_interior**2, interior_grid))
            )
            iid_taylor_terms.append(
                float(np.trapezoid(iid_taylor_interior**2, interior_grid))
            )
            tri_recon_errors.append(
                float(np.trapezoid(tri_recon_interior**2, interior_grid))
            )
            iid_recon_errors.append(
                float(np.trapezoid(iid_recon_interior**2, interior_grid))
            )

        tri_res = float(np.mean(tri_residuals))
        iid_res = float(np.mean(iid_residuals))
        tri_sup = float(np.mean(tri_sup_residuals))
        iid_sup = float(np.mean(iid_sup_residuals))
        tri_mse_mean = float(np.mean(tri_mse))
        iid_mse_mean = float(np.mean(iid_mse))
        tri_interior_mse_mean = float(np.mean(tri_interior_mse))
        iid_interior_mse_mean = float(np.mean(iid_interior_mse))
        tri_lin_mean = float(np.mean(tri_linear))
        iid_lin_mean = float(np.mean(iid_linear))
        tri_interior_res = float(np.mean(tri_interior_residuals))
        iid_interior_res = float(np.mean(iid_interior_residuals))
        tri_interior_sup = float(np.mean(tri_interior_sup_residuals))
        iid_interior_sup = float(np.mean(iid_interior_sup_residuals))
        tri_interior_lin_mean = float(np.mean(tri_interior_linear))
        iid_interior_lin_mean = float(np.mean(iid_interior_linear))
        tri_empirical_mean = float(np.mean(tri_empirical_terms))
        iid_empirical_mean = float(np.mean(iid_empirical_terms))
        tri_taylor_mean = float(np.mean(tri_taylor_terms))
        iid_taylor_mean = float(np.mean(iid_taylor_terms))
        tri_recon_mean = float(np.mean(tri_recon_errors))
        iid_recon_mean = float(np.mean(iid_recon_errors))

        summary_rows.append(
            {
                "experiment": "bahadur-residual-fixed-span",
                "n": n,
                "tri_residual": round(tri_res, 8),
                "iid_residual": round(iid_res, 8),
                "tri_mse": round(tri_mse_mean, 8),
                "iid_mse": round(iid_mse_mean, 8),
                "tri_residual_over_mse": round(tri_res / tri_mse_mean, 8),
                "iid_residual_over_mse": round(iid_res / iid_mse_mean, 8),
                "tri_sup_residual": round(tri_sup, 8),
                "iid_sup_residual": round(iid_sup, 8),
                "tri_linear_over_mse": round(tri_lin_mean / tri_mse_mean, 8),
                "iid_linear_over_mse": round(iid_lin_mean / iid_mse_mean, 8),
                "tri_interior_residual": round(tri_interior_res, 8),
                "iid_interior_residual": round(iid_interior_res, 8),
                "tri_interior_mse": round(tri_interior_mse_mean, 8),
                "iid_interior_mse": round(iid_interior_mse_mean, 8),
                "tri_interior_residual_over_mse": round(
                    tri_interior_res / tri_interior_mse_mean, 8
                ),
                "iid_interior_residual_over_mse": round(
                    iid_interior_res / iid_interior_mse_mean, 8
                ),
                "tri_interior_sup_residual": round(tri_interior_sup, 8),
                "iid_interior_sup_residual": round(iid_interior_sup, 8),
                "tri_interior_linear_over_mse": round(
                    tri_interior_lin_mean / tri_interior_mse_mean, 8
                ),
                "iid_interior_linear_over_mse": round(
                    iid_interior_lin_mean / iid_interior_mse_mean, 8
                ),
                "tri_empirical_term": round(tri_empirical_mean, 8),
                "iid_empirical_term": round(iid_empirical_mean, 8),
                "tri_taylor_term": round(tri_taylor_mean, 8),
                "iid_taylor_term": round(iid_taylor_mean, 8),
                "tri_reconstruction_error": round(tri_recon_mean, 10),
                "iid_reconstruction_error": round(iid_recon_mean, 10),
            }
        )
        curve_rows.extend(
            [
                {
                    "experiment": "bahadur-residual-fixed-span",
                    "setting": "triangular",
                    "n": n,
                    "value": round(tri_res, 8),
                },
                {
                    "experiment": "bahadur-residual-fixed-span",
                    "setting": "iid-mixture",
                    "n": n,
                    "value": round(iid_res, 8),
                },
            ]
        )
        tri_sup_curve.append(tri_sup)
        iid_sup_curve.append(iid_sup)
        tri_interior_residual_curve.append(tri_interior_res)
        iid_interior_residual_curve.append(iid_interior_res)
        tri_interior_sup_curve.append(tri_interior_sup)
        iid_interior_sup_curve.append(iid_interior_sup)
        tri_empirical_curve.append(tri_empirical_mean)
        iid_empirical_curve.append(iid_empirical_mean)
        tri_taylor_curve.append(tri_taylor_mean)
        iid_taylor_curve.append(iid_taylor_mean)
        tri_recon_error_curve.append(tri_recon_mean)
        iid_recon_error_curve.append(iid_recon_mean)

    tri_curve = np.asarray(
        [row["value"] for row in curve_rows if row["setting"] == "triangular"],
        dtype=float,
    )
    iid_curve = np.asarray(
        [row["value"] for row in curve_rows if row["setting"] == "iid-mixture"],
        dtype=float,
    )
    summary_rows.append(
        {
            "experiment": "bahadur-residual-rate",
            "tri_residual_rate": round(
                float(-np.polyfit(np.log(n_values), np.log(tri_curve), 1)[0]), 6
            ),
            "iid_residual_rate": round(
                float(-np.polyfit(np.log(n_values), np.log(iid_curve), 1)[0]), 6
            ),
            "tri_sup_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_sup_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_sup_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_sup_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "tri_interior_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_interior_residual_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_interior_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_interior_residual_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "tri_interior_sup_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_interior_sup_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_interior_sup_residual_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_interior_sup_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "tri_empirical_term_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_empirical_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_empirical_term_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_empirical_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "tri_taylor_term_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_taylor_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_taylor_term_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_taylor_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "tri_reconstruction_error_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(tri_recon_error_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
            "iid_reconstruction_error_rate": round(
                float(
                    -np.polyfit(
                        np.log(n_values),
                        np.log(np.asarray(iid_recon_error_curve, dtype=float)),
                        1,
                    )[0]
                ),
                6,
            ),
        }
    )

    return BahadurResidualResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bahadur residual diagnostics.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/glue_theorem_bahadur_residual"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bahadur_residual_research()
    export_rows_csv(
        result.summary_rows, args.csv_dir / "glue_theorem_bahadur_residual_summary.csv"
    )
    export_rows_csv(
        result.curve_rows, args.csv_dir / "glue_theorem_bahadur_residual_curves.csv"
    )


if __name__ == "__main__":
    main()
