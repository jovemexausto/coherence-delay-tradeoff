from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .glue_theorem_bahadur_rough_kernel import (
    empirical_cdf_at,
    fixed_span_means,
    sample_cusp_kernel,
    shifted_cusp_cdf,
    shifted_cusp_pdf,
    shifted_cusp_quantiles,
)


@dataclass(slots=True)
class BahadurRoughGrowthConfig:
    support_radius: float = 1.0
    growth_base_span: float = 0.2
    alpha_values: tuple[float, ...] = (1.0, 0.5, 0.25)
    growth_betas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    n_values: tuple[int, ...] = (32, 64, 128, 256)
    replications: int = 24
    quantile_grid_size: int = 96
    interior_epsilon: float = 0.1


@dataclass(slots=True)
class BahadurRoughGrowthResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def run_bahadur_rough_growth_research(
    config: BahadurRoughGrowthConfig | None = None,
) -> BahadurRoughGrowthResult:
    if config is None:
        config = BahadurRoughGrowthConfig()

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
        for beta in config.growth_betas:
            residual_curve: list[float] = []
            empirical_curve: list[float] = []
            taylor_curve: list[float] = []
            sup_curve: list[float] = []

            for n in n_values:
                span = config.growth_base_span * (n**beta)
                means = fixed_span_means(n, span, 1.0)
                q_target = shifted_cusp_quantiles(
                    quantile_grid, means, config.support_radius, alpha
                )
                f_target = shifted_cusp_pdf(
                    q_target, means, config.support_radius, alpha
                )
                safe_f = np.maximum(f_target, 1e-10)

                rep_residuals: list[float] = []
                rep_empirical: list[float] = []
                rep_taylor: list[float] = []
                rep_sup: list[float] = []

                for rep in range(config.replications):
                    rng = np.random.default_rng(
                        900_000 + 100 * n + rep + int(1000 * alpha) + int(100 * beta)
                    )
                    sample = means + config.support_radius * sample_cusp_kernel(
                        rng, n, alpha
                    )

                    qhat = np.quantile(sample, quantile_grid)
                    Fhat_qtarget = empirical_cdf_at(sample, q_target)
                    Fhat_qhat = empirical_cdf_at(sample, qhat)
                    Fbar_qhat = shifted_cusp_cdf(
                        qhat, means, config.support_radius, alpha
                    )

                    line = (quantile_grid - Fhat_qtarget) / safe_f
                    residual = qhat - q_target - line
                    empirical = (
                        -((Fhat_qhat - Fhat_qtarget) - (Fbar_qhat - quantile_grid))
                        / safe_f
                    )
                    taylor = (
                        -((Fbar_qhat - quantile_grid) - safe_f * (qhat - q_target))
                        / safe_f
                    )

                    rep_residuals.append(
                        float(np.trapezoid(residual**2, quantile_grid))
                    )
                    rep_empirical.append(
                        float(
                            np.trapezoid(empirical[interior_mask] ** 2, interior_grid)
                        )
                    )
                    rep_taylor.append(
                        float(np.trapezoid(taylor[interior_mask] ** 2, interior_grid))
                    )
                    rep_sup.append(float(np.max(np.abs(residual[interior_mask]))))

                residual_mean = float(np.mean(rep_residuals))
                empirical_mean = float(np.mean(rep_empirical))
                taylor_mean = float(np.mean(rep_taylor))
                sup_mean = float(np.mean(rep_sup))

                curve_rows.extend(
                    [
                        {
                            "experiment": "bahadur-rough-growth",
                            "alpha": round(float(alpha), 6),
                            "beta": round(float(beta), 6),
                            "setting": "triangular",
                            "n": n,
                            "value": round(residual_mean, 8),
                        },
                        {
                            "experiment": "bahadur-rough-growth",
                            "alpha": round(float(alpha), 6),
                            "beta": round(float(beta), 6),
                            "setting": "empirical",
                            "n": n,
                            "value": round(empirical_mean, 8),
                        },
                        {
                            "experiment": "bahadur-rough-growth",
                            "alpha": round(float(alpha), 6),
                            "beta": round(float(beta), 6),
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

            residual_rate = float(
                -np.polyfit(
                    np.log(n_values), np.log(np.asarray(residual_curve, dtype=float)), 1
                )[0]
            )
            empirical_rate = float(
                -np.polyfit(
                    np.log(n_values),
                    np.log(np.asarray(empirical_curve, dtype=float)),
                    1,
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

            summary_rows.append(
                {
                    "experiment": "bahadur-rough-growth-rate",
                    "alpha": round(float(alpha), 6),
                    "beta": round(float(beta), 6),
                    "residual_rate": round(residual_rate, 6),
                    "empirical_rate": round(empirical_rate, 6),
                    "taylor_rate": round(taylor_rate, 6),
                    "sup_rate": round(sup_rate, 6),
                    "last_emp_over_taylor": round(
                        empirical_curve[-1] / max(taylor_curve[-1], 1e-18), 6
                    ),
                }
            )

    return BahadurRoughGrowthResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run roughness-plus-growth Bahadur diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/glue_theorem_bahadur_rough_growth"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bahadur_rough_growth_research()
    export_rows_csv(
        result.summary_rows,
        args.csv_dir / "glue_theorem_bahadur_rough_growth_summary.csv",
    )
    export_rows_csv(
        result.curve_rows, args.csv_dir / "glue_theorem_bahadur_rough_growth_curves.csv"
    )


if __name__ == "__main__":
    main()
