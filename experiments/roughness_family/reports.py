from __future__ import annotations

import numpy as np

from .model import RoughnessScalingResult


def build_slope_rows(
    result: RoughnessScalingResult,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for index, H in enumerate(result.H_values):
        rows.append(
            {
                "H": f"{H:.2f}",
                "empirical_slope": round(float(result.fitted_slopes[index]), 3),
                "theory_slope": round(float(result.theory_slopes[index]), 3),
                "slope_error": round(
                    float(result.fitted_slopes[index] - result.theory_slopes[index]),
                    3,
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
