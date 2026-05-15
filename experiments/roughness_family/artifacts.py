from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import RoughnessScalingResult


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
                rf"$H={H:.2f}$ fit={result.fitted_slopes[h_index]:.2f}, "
                rf"theory={result.theory_slopes[h_index]:.2f}"
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

        low_branch = np.interp(
            np.log(optimal_window) - gap_grid,
            log_windows,
            errors,
            left=np.nan,
            right=np.nan,
        )
        high_branch = np.interp(
            np.log(optimal_window) + gap_grid,
            log_windows,
            errors,
            left=np.nan,
            right=np.nan,
        )
        branch_stack = np.vstack([low_branch, high_branch])
        excess_error = np.nanmean(branch_stack, axis=0) - optimal_error
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
