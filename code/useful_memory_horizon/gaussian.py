from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import export_rows_csv


@dataclass(slots=True)
class UCurveResult:
    drift_values: np.ndarray
    window_sizes: np.ndarray
    mean_error_grid: np.ndarray
    std_error_grid: np.ndarray
    empirical_n_star: np.ndarray
    empirical_e_min: np.ndarray
    slope: float
    scaled_constant: float


def _window_errors_for_drift(
    drift: float,
    window_sizes: np.ndarray,
    seeds: list[int],
    *,
    steps: int = 6000,
    process_scale: float = 0.1,
    observation_scale: float = 1.0,
) -> np.ndarray:
    errors = np.zeros((len(seeds), len(window_sizes)))
    tail_start = steps // 2
    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        mu = np.zeros(steps)
        for step in range(1, steps):
            mu[step] = mu[step - 1] + drift + rng.normal(scale=process_scale)
        obs = mu + rng.normal(scale=observation_scale, size=steps)
        for window_index, window in enumerate(window_sizes):
            estimate = np.full(steps, np.nan)
            for step in range(window - 1, steps):
                estimate[step] = np.mean(obs[step - window + 1 : step + 1])
            valid = np.arange(tail_start, steps)
            valid = valid[~np.isnan(estimate[tail_start:])]
            errors[seed_index, window_index] = float(
                np.mean(np.abs(mu[valid] - estimate[valid]))
            )
    return errors


def run_ucurve_experiment() -> UCurveResult:
    drift_values = np.asarray([0.001, 0.005, 0.01, 0.05], dtype=float)
    window_sizes = np.asarray([5, 10, 20, 50, 75, 100, 150, 200, 300, 500], dtype=int)
    seeds = list(range(20))
    error_grid = np.zeros((drift_values.size, window_sizes.size))
    std_grid = np.zeros((drift_values.size, window_sizes.size))
    empirical_n_star = np.zeros(drift_values.size, dtype=int)
    empirical_e_min = np.zeros(drift_values.size)

    for index, drift in enumerate(drift_values):
        errors = _window_errors_for_drift(
            drift, window_sizes, seeds, process_scale=0.0, observation_scale=1.0
        )
        mean_errors = np.mean(errors, axis=0)
        std_errors = np.std(errors, axis=0)
        error_grid[index] = mean_errors
        std_grid[index] = std_errors
        best = int(np.argmin(mean_errors))
        empirical_n_star[index] = int(window_sizes[best])
        empirical_e_min[index] = float(mean_errors[best])

    slope = float(np.polyfit(np.log(drift_values), np.log(empirical_e_min), 1)[0])
    scaled_constant = float(np.mean(empirical_e_min / np.cbrt(drift_values)))
    return UCurveResult(
        drift_values=drift_values,
        window_sizes=window_sizes,
        mean_error_grid=error_grid,
        std_error_grid=std_grid,
        empirical_n_star=empirical_n_star,
        empirical_e_min=empirical_e_min,
        slope=slope,
        scaled_constant=scaled_constant,
    )


def save_ucurve_figure(result: UCurveResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, drift in enumerate(result.drift_values):
        axes[0].plot(
            result.window_sizes,
            result.mean_error_grid[index],
            linewidth=1.4,
            marker="o",
            markersize=3,
            label=rf"$\zeta={drift:.3f}$",
        )
        best = int(np.argmin(result.mean_error_grid[index]))
        axes[0].scatter(
            result.window_sizes[best],
            result.mean_error_grid[index, best],
            color="black",
            s=28,
            zorder=4,
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Window size n")
    axes[0].set_ylabel("Mean absolute tracking error")
    axes[0].set_title("Lag-drift U-curve")
    axes[0].legend(loc="upper right")

    axes[1].plot(result.drift_values, result.empirical_e_min, marker="o", linewidth=1.5)
    fit_constant = np.exp(
        np.polyfit(np.log(result.drift_values), np.log(result.empirical_e_min), 1)[1]
    )
    axes[1].plot(
        result.drift_values,
        fit_constant * result.drift_values**result.slope,
        linestyle="--",
        color="0.35",
        label=rf"fit slope={result.slope:.3f}",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Drift rate $\zeta$")
    axes[1].set_ylabel(r"$\mathcal{E}_{min}$")
    axes[1].set_title("Minimum error vs drift")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_ucurve_rows(result: UCurveResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for drift, n_star, e_min in zip(
        result.drift_values,
        result.empirical_n_star,
        result.empirical_e_min,
        strict=True,
    ):
        rows.append(
            {
                "drift": round(float(drift), 4),
                "n_star": int(n_star),
                "e_min": round(float(e_min), 4),
                "scaled_constant": round(float(e_min / np.cbrt(drift)), 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gaussian U-curve artifacts.")
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/gaussian")
    )
    parser.add_argument("--csv-dir", type=Path, default=Path("artifacts/csv/gaussian"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    result = run_ucurve_experiment()
    save_ucurve_figure(result, args.figures_dir / "fig_ucurve.pdf")
    export_rows_csv(build_ucurve_rows(result), args.csv_dir / "gaussian_ucurve.csv")
