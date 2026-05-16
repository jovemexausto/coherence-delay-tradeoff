from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .glue_theorem_minimal import fixed_span_means, uniform_mixture_quantiles


@dataclass(slots=True)
class BahadurConfig:
    support_radius: float = 1.0
    fixed_span: float = 0.5
    n_values: tuple[int, ...] = (25, 50, 100, 200, 400)
    replications: int = 128
    quantile_grid_size: int = 256


@dataclass(slots=True)
class BahadurResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def sample_to_quantiles(sample: np.ndarray, quantile_grid: np.ndarray) -> np.ndarray:
    return np.quantile(sample, quantile_grid)


def triangular_and_iid_quantile_profiles(
    means: np.ndarray,
    support_radius: float,
    replications: int,
    quantile_grid: np.ndarray,
    n_seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_quantiles = uniform_mixture_quantiles(quantile_grid, means, support_radius)
    tri_profiles: list[np.ndarray] = []
    iid_profiles: list[np.ndarray] = []

    for rep in range(replications):
        tri_rng = np.random.default_rng(n_seed_base + 10_000 + rep)
        iid_rng = np.random.default_rng(n_seed_base + 20_000 + rep)
        tri_sample = means + tri_rng.uniform(
            -support_radius, support_radius, size=len(means)
        )
        iid_idx = iid_rng.integers(0, len(means), size=len(means))
        iid_sample = means[iid_idx] + iid_rng.uniform(
            -support_radius, support_radius, size=len(means)
        )
        tri_profiles.append(sample_to_quantiles(tri_sample, quantile_grid))
        iid_profiles.append(sample_to_quantiles(iid_sample, quantile_grid))

    return (
        target_quantiles,
        np.asarray(tri_profiles, dtype=float),
        np.asarray(iid_profiles, dtype=float),
        quantile_grid,
    )


def summarize_profiles(
    profiles: np.ndarray,
    target_quantiles: np.ndarray,
    quantile_grid: np.ndarray,
) -> tuple[float, float, float]:
    pointwise_mean = np.mean(profiles, axis=0)
    bias = pointwise_mean - target_quantiles
    centered = profiles - pointwise_mean
    variance = np.mean(centered**2, axis=0)
    mse = np.mean((profiles - target_quantiles) ** 2, axis=0)
    bias2_int = float(np.trapezoid(bias**2, quantile_grid))
    var_int = float(np.trapezoid(variance, quantile_grid))
    mse_int = float(np.trapezoid(mse, quantile_grid))
    return mse_int, bias2_int, var_int


def run_bahadur_research(
    config: BahadurConfig | None = None,
) -> BahadurResult:
    if config is None:
        config = BahadurConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    n_values = np.asarray(config.n_values, dtype=int)
    quantile_grid = (
        np.arange(config.quantile_grid_size, dtype=float) + 0.5
    ) / config.quantile_grid_size

    for n in n_values:
        means = fixed_span_means(n, config.fixed_span, 1.0)
        target_quantiles, tri_profiles, iid_profiles, grid = (
            triangular_and_iid_quantile_profiles(
                means,
                config.support_radius,
                config.replications,
                quantile_grid,
                n_seed_base=1000 * n,
            )
        )

        tri_mse, tri_bias2, tri_var = summarize_profiles(
            tri_profiles, target_quantiles, grid
        )
        iid_mse, iid_bias2, iid_var = summarize_profiles(
            iid_profiles, target_quantiles, grid
        )

        # Quantile-process identity on the discretized grid.
        tri_residual = tri_mse - (tri_bias2 + tri_var)
        iid_residual = iid_mse - (iid_bias2 + iid_var)

        curve_rows.extend(
            [
                {
                    "experiment": "bahadur-fixed-span",
                    "setting": "triangular",
                    "n": n,
                    "mse": round(tri_mse, 8),
                    "bias2": round(tri_bias2, 8),
                    "variance": round(tri_var, 8),
                    "residual": round(tri_residual, 12),
                },
                {
                    "experiment": "bahadur-fixed-span",
                    "setting": "iid-mixture",
                    "n": n,
                    "mse": round(iid_mse, 8),
                    "bias2": round(iid_bias2, 8),
                    "variance": round(iid_var, 8),
                    "residual": round(iid_residual, 12),
                },
            ]
        )

    tri_curve = np.asarray(
        [row["mse"] for row in curve_rows if row["setting"] == "triangular"],
        dtype=float,
    )
    iid_curve = np.asarray(
        [row["mse"] for row in curve_rows if row["setting"] == "iid-mixture"],
        dtype=float,
    )
    tri_bias2_curve = np.asarray(
        [row["bias2"] for row in curve_rows if row["setting"] == "triangular"],
        dtype=float,
    )
    tri_var_curve = np.asarray(
        [row["variance"] for row in curve_rows if row["setting"] == "triangular"],
        dtype=float,
    )
    iid_bias2_curve = np.asarray(
        [row["bias2"] for row in curve_rows if row["setting"] == "iid-mixture"],
        dtype=float,
    )
    iid_var_curve = np.asarray(
        [row["variance"] for row in curve_rows if row["setting"] == "iid-mixture"],
        dtype=float,
    )

    summary_rows.append(
        {
            "experiment": "bahadur-fixed-span",
            "tri_rate_a": round(
                float(-np.polyfit(np.log(n_values), np.log(tri_curve), 1)[0]), 6
            ),
            "iid_rate_a": round(
                float(-np.polyfit(np.log(n_values), np.log(iid_curve), 1)[0]), 6
            ),
            "tri_bias2_rate": round(
                float(
                    -np.polyfit(np.log(n_values), np.log(tri_bias2_curve + 1e-16), 1)[0]
                ),
                6,
            ),
            "tri_var_rate": round(
                float(-np.polyfit(np.log(n_values), np.log(tri_var_curve), 1)[0]), 6
            ),
            "iid_bias2_rate": round(
                float(
                    -np.polyfit(np.log(n_values), np.log(iid_bias2_curve + 1e-16), 1)[0]
                ),
                6,
            ),
            "iid_var_rate": round(
                float(-np.polyfit(np.log(n_values), np.log(iid_var_curve), 1)[0]), 6
            ),
            "tri_last_bias2_over_var": round(
                float(tri_bias2_curve[-1] / tri_var_curve[-1]), 6
            ),
            "iid_last_bias2_over_var": round(
                float(iid_bias2_curve[-1] / iid_var_curve[-1]), 6
            ),
        }
    )

    return BahadurResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bahadur-style glue theorem diagnostics."
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/glue_theorem_bahadur")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bahadur_research()
    export_rows_csv(
        result.summary_rows, args.csv_dir / "glue_theorem_bahadur_summary.csv"
    )
    export_rows_csv(result.curve_rows, args.csv_dir / "glue_theorem_bahadur_curves.csv")


if __name__ == "__main__":
    main()
