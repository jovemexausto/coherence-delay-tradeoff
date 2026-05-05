from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import Elec2ExperimentResult


def save_elec2_figure(result: Elec2ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detection = result.fixed_100
    time = np.arange(result.values.size)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(
        time, rolling_mean(detection.sigma, 100), color="tab:blue", linewidth=1.2
    )
    axes[0].axhline(
        result.config.warning_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="TCI threshold",
    )
    for warning in detection.warnings:
        axes[0].axvline(warning, color="tab:red", alpha=0.08, linewidth=0.8)
    for warning in result.adwin.warnings:
        axes[0].axvline(warning, color="tab:purple", alpha=0.05, linewidth=0.8)
    for event in result.events:
        axes[0].axvline(event, color="0.5", alpha=0.08, linewidth=0.8)
    axes[0].set_ylabel(r"$\hat\sigma_P$")
    axes[0].set_title("ELEC2 early-warning diagnostic (fixed n=100 vs ADWIN)")
    axes[0].plot([], [], color="tab:red", linewidth=1.0, label="TCI warnings")
    axes[0].plot([], [], color="tab:purple", linewidth=1.0, label="ADWIN warnings")
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].plot(
        time,
        rolling_mean(result.residual_signal, 50),
        color="tab:orange",
        linewidth=1.2,
        label="residual input",
    )
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time step")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def save_dynamic_nstar_figure(result: Elec2ExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.values.size)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

    axes[0].plot(
        time,
        rolling_mean(result.fixed_50.sigma, 100),
        label="fixed n=50",
        linewidth=1.1,
    )
    axes[0].plot(
        time,
        rolling_mean(result.fixed_300.sigma, 100),
        label="fixed n=300",
        linewidth=1.1,
    )
    axes[0].plot(
        time,
        rolling_mean(result.dynamic.sigma, 100),
        label="dynamic n*_t",
        linewidth=1.2,
    )
    for event in result.events:
        axes[0].axvline(event, color="0.7", alpha=0.05, linewidth=0.8)
    axes[0].set_ylabel(r"$\hat\sigma_P$")
    axes[0].set_title("Dynamic window adaptation on ELEC2")
    axes[0].legend(loc="lower left", ncol=3)

    axes[1].plot(time, result.dynamic.window_sizes, color="tab:green", linewidth=1.1)
    axes[1].set_ylabel(r"$n^*_t$")

    lead_data = [
        result.fixed_50.lead_times,
        result.fixed_300.lead_times,
        result.dynamic.lead_times,
        result.adwin.lead_times,
        result.cusum.lead_times,
        result.rls.lead_times,
        result.kalman.lead_times,
        result.frechet.lead_times,
    ]
    axes[2].boxplot(
        lead_data,
        labels=[
            "fixed 50",
            "fixed 300",
            "dynamic",
            "ADWIN",
            "CUSUM",
            "FF-RLS",
            "Kalman",
            "Fr'echet",
        ],
        showfliers=False,
    )
    axes[2].set_ylabel("Lead time")
    axes[2].set_xlabel("Strategy")

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
