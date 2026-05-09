from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import BikesExperimentResult
from .reports import ARENA_BASELINES, summarize_detection


def save_bikes_figure(result: BikesExperimentResult, output_path: Path) -> None:
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
    axes[0].set_title("Bikes early-warning diagnostic (fixed n=100 vs ADWIN)")
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


def save_dynamic_nstar_figure(result: BikesExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.values.size)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    pairs = [
        (
            "EWMA",
            result.arena["ewma"].window_sizes,
            result.arena["ewma_umr"].window_sizes,
        ),
        (
            "Window Dilemma",
            result.arena["window_dilemma"].window_sizes,
            result.arena["window_dilemma_umr"].window_sizes,
        ),
        (
            "MELO-style",
            result.arena["melo"].window_sizes,
            result.arena["melo_umr"].window_sizes,
        ),
        ("ADWIN", result.adwin.window_sizes, result.adwin_umr.window_sizes),
    ]
    for axis, (name, base_series, regulated_series) in zip(axes, pairs, strict=True):
        axis.plot(time, base_series, linewidth=1.0, label=name)
        axis.plot(time, regulated_series, linewidth=1.2, label=f"{name} + UMR")
        axis.set_ylabel("Horizon")
        axis.set_title(f"{name} vs {name} + UMR")
        axis.legend(loc="upper right", ncol=2)
        axis.grid(alpha=0.2, linewidth=0.5)

    axes[-1].set_xlabel("Time step")

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def save_umr_arena_figure(result: BikesExperimentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [baseline.replace("_", " ") for baseline in ARENA_BASELINES]
    base_precision: list[float] = []
    umr_precision: list[float] = []
    base_leads: list[float] = []
    umr_leads: list[float] = []

    for baseline in ARENA_BASELINES:
        base_summary = summarize_detection(result.arena[baseline])
        umr_summary = summarize_detection(result.arena[f"{baseline}_umr"])
        base_precision.append(base_summary["precision"])
        umr_precision.append(umr_summary["precision"])
        base_leads.append(base_summary["median_lead"])
        umr_leads.append(umr_summary["median_lead"])

    x = np.arange(len(ARENA_BASELINES))
    width = 0.36
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

    axes[0].bar(x - width / 2, base_precision, width=width, label="base")
    axes[0].bar(x + width / 2, umr_precision, width=width, label="base + UMR")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Bikes arena: backend vs backend + UMR")
    axes[0].legend(loc="upper left")

    axes[1].bar(x - width / 2, base_leads, width=width, label="base")
    axes[1].bar(x + width / 2, umr_leads, width=width, label="base + UMR")
    axes[1].set_ylabel("Median lead")

    time = np.arange(result.values.size)
    axes[2].plot(
        time, result.adwin.window_sizes, label="ADWIN", linewidth=1.0, color="tab:gray"
    )
    axes[2].plot(
        time,
        result.adwin_umr.window_sizes,
        label="ADWIN + UMR",
        linewidth=1.2,
        color="tab:green",
    )
    axes[2].set_ylabel("Horizon")
    axes[2].set_title("ADWIN vs ADWIN + UMR horizon")
    axes[2].set_xlabel("Time step")
    axes[2].legend(loc="upper right", ncol=2)

    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
