from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class RoughnessScalingConfig:
    H_values: tuple[float, ...] = (0.5, 0.75, 1.0)
    zeta_values: tuple[float, ...] = (0.003, 0.005, 0.008, 0.012, 0.018, 0.027)
    window_sizes: tuple[int, ...] = field(
        default_factory=lambda: tuple(
            np.unique(np.round(np.geomspace(1, 2200, 160)).astype(int)).tolist()
        )
    )
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
        errors[index] = float(np.mean(np.abs(noise + bias)))
    return errors


def run_roughness_scaling_experiment(
    config: RoughnessScalingConfig | None = None,
) -> RoughnessScalingResult:
    cfg = config or RoughnessScalingConfig()
    H_values = np.asarray(cfg.H_values, dtype=float)
    zeta_values = np.asarray(cfg.zeta_values, dtype=float)
    window_sizes = np.asarray(cfg.window_sizes, dtype=float)

    n_h = H_values.size
    n_z = zeta_values.size
    n_w = window_sizes.size
    per_seed = np.zeros((len(cfg.seeds), n_h, n_z, n_w), dtype=float)

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
    fitted_slopes = np.zeros(n_h, dtype=float)
    theory_intercepts = np.zeros(n_h, dtype=float)
    theory_window_grid = np.zeros((n_h, n_z), dtype=float)

    log_zeta = np.log(zeta_values)
    for h_index, slope in enumerate(theory_slopes):
        log_optimal = np.log(optimal_windows[h_index])
        fitted_slopes[h_index] = float(np.polyfit(log_zeta, log_optimal, 1)[0])
        intercept = float(np.mean(log_optimal - slope * log_zeta))
        theory_intercepts[h_index] = intercept
        theory_window_grid[h_index] = np.exp(intercept) * zeta_values**slope

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
