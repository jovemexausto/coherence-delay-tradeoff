from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import (
    SampleComplexityResult,
    TGT_COLORS,
    TGT_CONDITIONS,
    TGT_LABELS,
    TGTResult,
    SinkhornRuntimeResult,
    UCurveResult,
)


def save_ablation_figure(results: dict[str, TGTResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for condition in TGT_CONDITIONS:
        result = results[condition]
        axes[0].plot(
            time,
            rolling_mean(result.v_total, 20),
            color=TGT_COLORS[condition],
            linewidth=1.3,
            label=TGT_LABELS[condition],
        )
        axes[1].plot(
            time,
            rolling_mean(result.tci, 20),
            color=TGT_COLORS[condition],
            linewidth=1.3,
            label=TGT_LABELS[condition],
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$V_{total}$")
    axes[0].set_title("Component ablation under passive drift")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower left", ncol=2)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
    fit_curve = fit_constant * result.drift_values**result.slope
    axes[1].plot(
        result.drift_values,
        fit_curve,
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


def save_sigma_p_complexity_figure(
    result: SampleComplexityResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    axis.plot(
        result.window_sizes,
        result.mean_absolute_error,
        marker="o",
        linewidth=1.5,
        color="tab:blue",
        label="measured MAE",
    )
    fit_constant = np.exp(
        np.polyfit(np.log(result.window_sizes), np.log(result.mean_absolute_error), 1)[
            1
        ]
    )
    fit_curve = fit_constant * result.window_sizes**result.slope
    axis.plot(
        result.window_sizes,
        fit_curve,
        linestyle="--",
        color="0.35",
        label=rf"fit slope={result.slope:.3f}",
    )
    axis.fill_between(
        result.window_sizes,
        result.mean_absolute_error - result.std_absolute_error,
        result.mean_absolute_error + result.std_absolute_error,
        color="tab:blue",
        alpha=0.15,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Window size n")
    axis.set_ylabel(r"MAE $|\sigma_P - \hat\sigma_P|$")
    axis.set_title("Variance-dominated sample complexity")
    axis.grid(alpha=0.2, linewidth=0.5)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sinkhorn_figure(result: SampleComplexityResult, output_path: Path) -> None:
    save_sigma_p_complexity_figure(result, output_path)


def save_sinkhorn_runtime_figure(
    result: SinkhornRuntimeResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for d_index, dimension in enumerate(result.dimensions):
        for e_index, epsilon in enumerate(result.epsilons):
            axes[0].plot(
                result.window_sizes,
                result.mean_runtime_ms[d_index, :, e_index],
                marker="o",
                linewidth=1.3,
                label=rf"d={dimension}, $\varepsilon$={epsilon:.2f}",
            )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Window size n")
    axes[0].set_ylabel("Mean runtime (ms)")
    axes[0].set_title("Sinkhorn runtime scaling")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)

    for d_index, dimension in enumerate(result.dimensions):
        axes[1].plot(
            result.epsilons,
            result.mean_abs_bias[d_index, -1, :],
            marker="o",
            linewidth=1.3,
            label=rf"d={dimension}, n={result.window_sizes[-1]}",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Regularization $\varepsilon$")
    axes[1].set_ylabel("Mean absolute bias")
    axes[1].set_title("Bias vs regularization")
    axes[1].legend(loc="upper right")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
