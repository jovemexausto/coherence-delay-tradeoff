from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from .common import export_rows_csv


def _default_window_sizes() -> tuple[int, ...]:
    values = np.unique(np.round(np.geomspace(1, 2200, 160)).astype(int)).tolist()
    return tuple(int(value) for value in values)


@dataclass(slots=True)
class RoughnessScalingConfig:
    H_values: tuple[float, ...] = (0.5, 0.75, 1.0)
    zeta_values: tuple[float, ...] = (0.003, 0.005, 0.008, 0.012, 0.018, 0.027)
    window_sizes: tuple[int, ...] = field(default_factory=_default_window_sizes)
    seeds: tuple[int, ...] = tuple(range(12))
    replicas: int = 4000
    noise_scale: float = 1.0
    bias_scale: float = 1.0
    reference_zeta_index: int = 2


@dataclass(slots=True)
class RoughnessScalingResult:
    config: RoughnessScalingConfig
    H_values: np.ndarray
    zeta_values: np.ndarray
    window_sizes: np.ndarray
    mean_error_grid: np.ndarray
    std_error_grid: np.ndarray
    optimal_windows: np.ndarray
    optimal_errors: np.ndarray
    fitted_slopes: np.ndarray
    theory_slopes: np.ndarray
    theory_intercepts: np.ndarray
    theory_window_grid: np.ndarray


def _simulate_error_curve(
    *,
    H: float,
    zeta: float,
    window_sizes: np.ndarray,
    seed: int,
    replicas: int,
    noise_scale: float,
    bias_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    errors = np.zeros(window_sizes.size, dtype=float)
    for index, window in enumerate(window_sizes):
        noise = rng.normal(scale=noise_scale / np.sqrt(window), size=replicas)
        bias = bias_scale * zeta * window**H
        sample_errors = cast(np.ndarray, np.abs(noise + bias))
        errors[index] = float(np.mean(sample_errors))
    return errors


def run_roughness_scaling_experiment(
    config: RoughnessScalingConfig | None = None,
) -> RoughnessScalingResult:
    cfg = config or RoughnessScalingConfig()
    H_values = np.asarray(cfg.H_values, dtype=float)
    zeta_values = np.asarray(cfg.zeta_values, dtype=float)
    window_sizes = np.asarray(cfg.window_sizes, dtype=float)
    per_seed = np.zeros(
        (len(cfg.seeds), H_values.size, zeta_values.size, window_sizes.size),
        dtype=float,
    )

    for seed_index, seed in enumerate(cfg.seeds):
        for h_index, H in enumerate(H_values):
            for z_index, zeta in enumerate(zeta_values):
                local_seed = int(100_000 * seed + 1_000 * h_index + 10 * z_index)
                per_seed[seed_index, h_index, z_index] = _simulate_error_curve(
                    H=H,
                    zeta=zeta,
                    window_sizes=window_sizes,
                    seed=local_seed,
                    replicas=cfg.replicas,
                    noise_scale=cfg.noise_scale,
                    bias_scale=cfg.bias_scale,
                )

    mean_error_grid = np.mean(per_seed, axis=0)
    std_error_grid = np.std(per_seed, axis=0)
    best_indices = np.argmin(mean_error_grid, axis=-1)
    optimal_windows = window_sizes[best_indices]
    optimal_errors = np.take_along_axis(
        mean_error_grid, best_indices[..., None], axis=-1
    ).squeeze(-1)
    theory_slopes = -2.0 / (1.0 + 2.0 * H_values)
    fitted_slopes = np.zeros(H_values.size, dtype=float)
    theory_intercepts = np.zeros(H_values.size, dtype=float)
    theory_window_grid = np.zeros((H_values.size, zeta_values.size), dtype=float)

    log_zeta = np.log(zeta_values)
    for h_index, slope in enumerate(theory_slopes):
        log_optimal = np.log(optimal_windows[h_index])
        fit_coefficients = cast(np.ndarray, np.polyfit(log_zeta, log_optimal, 1))
        fitted_slopes[h_index] = float(fit_coefficients[0])
        centered_log_optimal = cast(np.ndarray, log_optimal - slope * log_zeta)
        intercept = float(np.mean(centered_log_optimal))
        theory_intercepts[h_index] = intercept
        theory_window_grid[h_index] = cast(
            np.ndarray, np.exp(intercept) * zeta_values**slope
        )

    return RoughnessScalingResult(
        config=cfg,
        H_values=H_values,
        zeta_values=zeta_values,
        window_sizes=window_sizes,
        mean_error_grid=mean_error_grid,
        std_error_grid=std_error_grid,
        optimal_windows=optimal_windows,
        optimal_errors=optimal_errors,
        fitted_slopes=fitted_slopes,
        theory_slopes=theory_slopes,
        theory_intercepts=theory_intercepts,
        theory_window_grid=theory_window_grid,
    )


def save_roughness_scaling_figure(
    result: RoughnessScalingResult, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_index = min(
        max(result.config.reference_zeta_index, 0), len(result.zeta_values) - 1
    )
    reference_zeta = float(result.zeta_values[reference_index])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for h_index, H in enumerate(result.H_values):
        errors = result.mean_error_grid[h_index, reference_index]
        best_index = int(errors.argmin())
        line = axes[0].plot(
            result.window_sizes,
            errors,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=rf"$H={H:.2f}$",
        )[0]
        axes[0].scatter(
            [result.window_sizes[best_index]],
            [errors[best_index]],
            color="black",
            s=24,
            zorder=4,
        )
        axes[0].scatter(
            [result.window_sizes[-1]],
            [errors[-1]],
            color=line.get_color(),
            edgecolors="black",
            linewidths=0.6,
            s=30,
            zorder=5,
        )

    for h_index, H in enumerate(result.H_values):
        line = axes[1].plot(
            result.zeta_values,
            result.optimal_windows[h_index],
            linewidth=1.5,
            marker="o",
            label=(
                rf"$H={H:.2f}$ fit={result.fitted_slopes[h_index]:.2f}, theory={result.theory_slopes[h_index]:.2f}"
            ),
        )[0]
        axes[1].plot(
            result.zeta_values,
            result.theory_window_grid[h_index],
            linestyle="--",
            linewidth=1.0,
            color=line.get_color(),
        )
        axes[1].scatter(
            [result.zeta_values[-1]],
            [result.optimal_windows[h_index, -1]],
            color=line.get_color(),
            edgecolors="black",
            linewidths=0.6,
            s=34,
            zorder=5,
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(float(result.window_sizes[0]), float(result.window_sizes[-1]))
    axes[0].set_xlabel("Window size n")
    axes[0].set_ylabel("Mean absolute tracking error")
    axes[0].set_title(rf"Finite optimum at fixed $\zeta={reference_zeta:.3f}$")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(float(result.zeta_values[0]), float(result.zeta_values[-1]))
    axes[1].set_xlabel(r"Roughness scale $\zeta$")
    axes[1].set_ylabel("Empirical optimal horizon")
    axes[1].set_title("Roughness-indexed horizon scaling")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_horizon_misalignment_figure(
    result: RoughnessScalingResult, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_index = min(
        max(result.config.reference_zeta_index, 0), len(result.zeta_values) - 1
    )
    reference_zeta = float(result.zeta_values[reference_index])
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    common_gap_limit = min(
        min(
            abs(
                np.log(
                    float(result.window_sizes[0])
                    / float(result.optimal_windows[h_index, reference_index])
                )
            ),
            abs(
                np.log(
                    float(result.window_sizes[-1])
                    / float(result.optimal_windows[h_index, reference_index])
                )
            ),
        )
        for h_index in range(len(result.H_values))
    )
    common_gap_limit = max(0.0, common_gap_limit - 1e-3)
    gap_grid = np.linspace(0.0, common_gap_limit, 121)
    for h_index, H in enumerate(result.H_values):
        optimal_window = float(result.optimal_windows[h_index, reference_index])
        optimal_error = float(result.optimal_errors[h_index, reference_index])
        log_windows = np.log(result.window_sizes)
        errors = result.mean_error_grid[h_index, reference_index]
        low_branch = cast(
            np.ndarray,
            np.interp(
                np.log(optimal_window) - gap_grid,
                log_windows,
                errors,
                left=np.nan,
                right=np.nan,
            ),
        )
        high_branch = cast(
            np.ndarray,
            np.interp(
                np.log(optimal_window) + gap_grid,
                log_windows,
                errors,
                left=np.nan,
                right=np.nan,
            ),
        )
        excess_error = (
            cast(
                np.ndarray,
                np.nanmean(np.vstack((low_branch, high_branch)), axis=0),
            )
            - optimal_error
        )
        valid = np.isfinite(excess_error)
        axis.plot(
            gap_grid[valid],
            excess_error[valid],
            linewidth=1.5,
            marker="o",
            markersize=3,
            markevery=10,
            label=rf"$H={H:.2f}$",
        )

    axis.set_xlabel(r"Relative horizon gap $|\log(n / n^*)|$")
    axis.set_ylabel("Excess tracking error")
    axis.set_title(rf"Misalignment cost at $\zeta={reference_zeta:.3f}$")
    axis.set_xlim(0.0, common_gap_limit)
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_slope_rows(result: RoughnessScalingResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, H in enumerate(result.H_values):
        rows.append(
            {
                "H": f"{H:.2f}",
                "empirical_slope": round(float(result.fitted_slopes[index]), 3),
                "theory_slope": round(float(result.theory_slopes[index]), 3),
                "slope_error": round(
                    float(result.fitted_slopes[index] - result.theory_slopes[index]), 3
                ),
                "median_optimal_window": round(
                    float(result.optimal_windows[index].mean()), 2
                ),
            }
        )
    return rows


def build_optimal_window_rows(
    result: RoughnessScalingResult,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for h_index, H in enumerate(result.H_values):
        for z_index, zeta in enumerate(result.zeta_values):
            rows.append(
                {
                    "H": f"{H:.2f}",
                    "zeta": round(float(zeta), 5),
                    "optimal_window": round(
                        float(result.optimal_windows[h_index, z_index]), 2
                    ),
                    "optimal_error": round(
                        float(result.optimal_errors[h_index, z_index]), 5
                    ),
                    "theory_window": round(
                        float(result.theory_window_grid[h_index, z_index]), 2
                    ),
                }
            )
    return rows


def build_misalignment_rows(
    result: RoughnessScalingResult,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    reference_index = min(
        max(result.config.reference_zeta_index, 0), len(result.zeta_values) - 1
    )
    for h_index, H in enumerate(result.H_values):
        optimal_window = float(result.optimal_windows[h_index, reference_index])
        optimal_error = float(result.optimal_errors[h_index, reference_index])
        for window, error in zip(
            result.window_sizes,
            result.mean_error_grid[h_index, reference_index],
            strict=True,
        ):
            rows.append(
                {
                    "H": f"{H:.2f}",
                    "window": round(float(window), 2),
                    "relative_gap": round(
                        abs(float(np.log(window / optimal_window))), 5
                    ),
                    "excess_error": round(float(error - optimal_error), 5),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate roughness-indexed horizon scaling artifacts."
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("artifacts/figures/roughness_family")
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/roughness_family")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    result = run_roughness_scaling_experiment(RoughnessScalingConfig())
    save_roughness_scaling_figure(result, args.figures_dir / "fig_roughness_family.pdf")
    save_horizon_misalignment_figure(
        result, args.figures_dir / "fig_horizon_misalignment.pdf"
    )
    export_rows_csv(
        build_slope_rows(result), args.csv_dir / "roughness_family_slopes.csv"
    )
    export_rows_csv(
        build_optimal_window_rows(result), args.csv_dir / "roughness_family_optima.csv"
    )
    export_rows_csv(
        build_misalignment_rows(result),
        args.csv_dir / "roughness_family_misalignment.csv",
    )
