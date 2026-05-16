from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .common import export_rows_csv
from .glue_theorem_research import (
    fixed_span_means,
    fit_power_law_exponent,
)


@dataclass(slots=True)
class MinimalGlueConfig:
    support_radius: float = 1.0
    fixed_span: float = 0.5
    n_values: tuple[int, ...] = (25, 50, 100, 200, 400)
    replications: int = 96
    quantile_grid_size: int = 512


@dataclass(slots=True)
class MinimalGlueResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def uniform_mixture_cdf(x_value: float, means: np.ndarray, radius: float) -> float:
    left = means - radius
    right = means + radius
    return float(np.mean(np.clip((x_value - left) / (2.0 * radius), 0.0, 1.0)))


def uniform_mixture_quantiles(
    quantile_grid: np.ndarray,
    means: np.ndarray,
    radius: float,
) -> np.ndarray:
    bracket = float(np.max(np.abs(means)) + radius + 1.0)
    lower = -bracket
    upper = bracket
    quantiles: list[float] = []
    for u in quantile_grid:
        root = float(
            brentq(  # pyright: ignore[reportArgumentType]
                lambda x_value, target=u: (
                    uniform_mixture_cdf(x_value, means, radius) - target
                ),
                lower,
                upper,
                xtol=1e-10,
            )
        )
        quantiles.append(root)
    return np.asarray(quantiles, dtype=float)


def sample_to_quantile_target(
    sample: np.ndarray,
    target_quantiles: np.ndarray,
    quantile_grid: np.ndarray,
) -> float:
    sample_q = np.quantile(sample, quantile_grid)
    return float(np.sqrt(np.mean((sample_q - target_quantiles) ** 2)))


def run_minimal_glue_research(
    config: MinimalGlueConfig | None = None,
) -> MinimalGlueResult:
    if config is None:
        config = MinimalGlueConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    n_values = np.asarray(config.n_values, dtype=int)
    quantile_grid = (
        np.arange(config.quantile_grid_size, dtype=float) + 0.5
    ) / config.quantile_grid_size

    for n in n_values:
        means = fixed_span_means(n, config.fixed_span, 1.0)
        target_quantiles = uniform_mixture_quantiles(
            quantile_grid, means, config.support_radius
        )

        tri_values: list[float] = []
        iid_values: list[float] = []
        for rep in range(config.replications):
            tri_rng = np.random.default_rng(100_000 + 100 * n + rep)
            iid_rng = np.random.default_rng(200_000 + 100 * n + rep)
            tri_sample = means + tri_rng.uniform(
                -config.support_radius, config.support_radius, size=n
            )
            iid_idx = iid_rng.integers(0, n, size=n)
            iid_sample = means[iid_idx] + iid_rng.uniform(
                -config.support_radius, config.support_radius, size=n
            )
            tri_values.append(
                sample_to_quantile_target(tri_sample, target_quantiles, quantile_grid)
            )
            iid_values.append(
                sample_to_quantile_target(iid_sample, target_quantiles, quantile_grid)
            )

        tri_mean = float(np.mean(tri_values))
        iid_mean = float(np.mean(iid_values))
        curve_rows.extend(
            [
                {
                    "experiment": "bounded-support-fixed-span",
                    "setting": "triangular",
                    "n": n,
                    "value": round(tri_mean, 6),
                },
                {
                    "experiment": "bounded-support-fixed-span",
                    "setting": "iid-mixture",
                    "n": n,
                    "value": round(iid_mean, 6),
                },
            ]
        )

    tri_curve = np.asarray(
        [row["value"] for row in curve_rows if row["setting"] == "triangular"],
        dtype=float,
    )
    iid_curve = np.asarray(
        [row["value"] for row in curve_rows if row["setting"] == "iid-mixture"],
        dtype=float,
    )
    tri_rate = fit_power_law_exponent(n_values.astype(float), tri_curve)
    iid_rate = fit_power_law_exponent(n_values.astype(float), iid_curve)
    summary_rows.append(
        {
            "experiment": "bounded-support-fixed-span",
            "support_radius": config.support_radius,
            "span": config.fixed_span,
            "tri_rate_a": round(tri_rate, 6),
            "iid_rate_a": round(iid_rate, 6),
            "rate_gap": round(abs(tri_rate - iid_rate), 6),
            "tri_over_iid_last": round(float(tri_curve[-1] / iid_curve[-1]), 6),
        }
    )

    return MinimalGlueResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal glue theorem research sweeps."
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/glue_theorem_minimal")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_minimal_glue_research()
    export_rows_csv(
        result.summary_rows, args.csv_dir / "glue_theorem_minimal_summary.csv"
    )
    export_rows_csv(result.curve_rows, args.csv_dir / "glue_theorem_minimal_curves.csv")


if __name__ == "__main__":
    main()
