from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from ..core.types import SummaryRow
from .reports import summarize_result
from .model import (
    ABLATION_CONDITIONS,
    TPTActiveBenchmarkResult,
    TPTResult,
    _condition_label,
)


def save_tcie_calibration_figure(
    rows: Sequence[SummaryRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lambdas = sorted({float(row["lambda"]) for row in rows})
    thresholds = sorted({float(row["threshold"]) for row in rows})
    masking_delay = np.full((len(lambdas), len(thresholds)), np.nan)
    collapse_delay = np.full((len(lambdas), len(thresholds)), np.nan)
    healthy_fp = np.full((len(lambdas), len(thresholds)), np.nan)

    lambda_index = {value: index for index, value in enumerate(lambdas)}
    threshold_index = {value: index for index, value in enumerate(thresholds)}
    for row in rows:
        i = lambda_index[float(row["lambda"])]
        j = threshold_index[float(row["threshold"])]
        masking_delay[i, j] = (
            np.nan
            if row["masking_median_delay"] == "NA"
            else float(row["masking_median_delay"])
        )
        collapse_delay[i, j] = (
            np.nan
            if row["collapse_median_delay"] == "NA"
            else float(row["collapse_median_delay"])
        )
        healthy_fp[i, j] = float(row["mean_healthy_false_positives"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    panels = [
        (masking_delay, "Masking median delay"),
        (collapse_delay, "Collapse median delay"),
        (healthy_fp, "Healthy false positives / run"),
    ]

    for axis, (matrix, title) in zip(axes, panels, strict=True):
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xticks(
            np.arange(len(thresholds)), [f"{value:.2f}" for value in thresholds]
        )
        axis.set_yticks(np.arange(len(lambdas)), [f"{value:.1f}" for value in lambdas])
        axis.set_xlabel("Threshold")
        axis.set_ylabel(r"$\lambda$")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isnan(value):
                    continue
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < np.nanmedian(matrix) else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("CI^E calibration on the active benchmark")
    fig.savefig(output_path)
    plt.close(fig)


def save_masking_grid_figure(
    summary_rows: Sequence[SummaryRow],
    influences: list[float],
    lambdas: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coercive_rows = [row for row in summary_rows if row["regime"] == "coercive"]
    influence_index = {value: index for index, value in enumerate(influences)}
    lambda_index = {value: index for index, value in enumerate(lambdas)}
    mean_tci = np.full((len(lambdas), len(influences)), np.nan)
    mean_tcie = np.full((len(lambdas), len(influences)), np.nan)
    masking_gap = np.full((len(lambdas), len(influences)), np.nan)

    for row in coercive_rows:
        i = lambda_index[float(row["lambda"])]
        j = influence_index[float(row["influence"])]
        mean_tci[i, j] = float(row["mean_tail_tci"])
        mean_tcie[i, j] = float(row["mean_tail_tcie"])
        masking_gap[i, j] = float(row["mean_masking_gap"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    panels = [
        (mean_tci, "Coercive mean CI"),
        (mean_tcie, "Coercive mean CI^E"),
        (masking_gap, "Masking gap: CI - CI^E"),
    ]

    for axis, (matrix, title) in zip(axes, panels, strict=True):
        image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xticks(
            np.arange(len(influences)), [f"{value:.1f}" for value in influences]
        )
        axis.set_yticks(np.arange(len(lambdas)), [f"{value:.1f}" for value in lambdas])
        axis.set_xlabel("Influence")
        axis.set_ylabel(r"$\lambda$")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < 0.45 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(output_path)
    plt.close(fig)


def save_active_benchmark_figure(
    result: TPTActiveBenchmarkResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.config.steps)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(
        time, result.latent_state, color="black", linewidth=1.3, label="latent"
    )
    axes[0].plot(
        time,
        result.uncontrolled_state,
        color="0.6",
        linewidth=1.0,
        linestyle="--",
        label="free state",
    )
    axes[0].plot(
        time,
        result.posterior_mean,
        color="tab:blue",
        linewidth=1.2,
        label="posterior mean",
    )
    axes[0].axvline(result.config.masking_start, color="tab:orange", linestyle="--")
    axes[0].axvline(result.config.collapse_start, color="tab:red", linestyle="--")
    axes[0].set_ylabel("State")
    axes[0].set_title("Active benchmark: healthy -> coercive masking -> collapse")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(
        time,
        result.adwin_signal,
        color="tab:purple",
        linewidth=1.1,
        label="ADWIN input",
    )
    for warning in result.adwin_warnings:
        axes[1].axvline(warning, color="tab:purple", alpha=0.08, linewidth=0.8)
    axes[1].set_ylabel("Observation")
    axes[1].legend(loc="upper left")

    axes[2].plot(
        time, result.action_gap, color="tab:brown", linewidth=1.1, label="action gap"
    )
    axes[2].plot(
        time,
        result.effort_signal,
        color="tab:pink",
        linewidth=1.3,
        label="coercive effort",
    )
    axes[2].set_ylabel("Effort")
    axes[2].legend(loc="upper left", ncol=2)

    axes[3].plot(
        time, result.tci, color="0.45", linewidth=1.0, linestyle="--", label="CI"
    )
    axes[3].plot(
        time,
        result.tcie,
        color="tab:red",
        linewidth=1.4,
        label="CI^E",
    )
    axes[3].axhline(
        result.config.tci_threshold, color="0.5", linestyle=":", linewidth=0.9
    )
    axes[3].axhline(
        result.config.tcie_threshold, color="tab:red", linestyle=":", linewidth=0.9
    )
    for warning in result.tci_warnings:
        axes[3].axvline(warning, color="0.6", alpha=0.08, linewidth=0.8)
    for warning in result.tcie_warnings:
        axes[3].axvline(warning, color="tab:red", alpha=0.08, linewidth=0.8)
    axes[3].set_ylabel("CI")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].legend(loc="lower left", ncol=2)

    for axis in axes:
        axis.axvline(
            result.config.masking_start,
            color="tab:orange",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axvline(
            result.config.collapse_start, color="tab:red", linestyle="--", linewidth=0.9
        )
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_particle_tracking_figure(result: TPTResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.config.steps)
    band = 2.0 * result.posterior_std

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(
        time, result.latent_state, label="latent state", color="black", linewidth=1.4
    )
    if result.config.influence > 0:
        axes[0].plot(
            time,
            result.uncontrolled_state,
            label="uncontrolled state",
            color="0.55",
            linewidth=1.0,
            linestyle="--",
        )
    axes[0].plot(
        time,
        result.posterior_mean,
        label="posterior mean",
        color="tab:blue",
        linewidth=1.3,
    )
    axes[0].fill_between(
        time,
        result.posterior_mean - band,
        result.posterior_mean + band,
        color="tab:blue",
        alpha=0.16,
        label="posterior +/- 2 std",
    )
    axes[0].scatter(
        time,
        result.observations,
        s=10,
        alpha=0.28,
        color="tab:orange",
        label="observations",
    )
    axes[0].set_ylabel("State")
    axes[0].set_title(
        f"Triadic Particle Tracker: {_condition_label(result.condition)} (influence={result.config.influence:.2f})"
    )
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].plot(time, result.ess, label="ESS / N", color="tab:green", linewidth=1.2)
    axes[1].plot(
        time, result.entropy, label="weight entropy", color="tab:purple", linewidth=1.2
    )
    axes[1].axhline(
        result.config.resample_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="resample threshold",
    )
    resampled_steps = time[result.resampled]
    if resampled_steps.size:
        axes[1].vlines(
            resampled_steps, 0.0, 1.0, color="tab:red", alpha=0.08, linewidth=0.8
        )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Convergence")
    axes[1].legend(loc="lower left", ncol=3)

    axes[2].plot(
        time,
        result.action_gap,
        label="action gap",
        color="tab:brown",
        linewidth=1.1,
    )
    axes[2].plot(
        time,
        result.effort_signal,
        label="coercive effort",
        color="tab:pink",
        linewidth=1.3,
    )
    axes[2].set_ylabel("Effort")
    axes[2].legend(loc="upper left", ncol=2)

    axes[3].plot(
        time, result.sigma_p, label=r"$\sigma_P$", color="tab:blue", linewidth=1.0
    )
    axes[3].plot(
        time,
        result.sigma_p_eff,
        label=r"$\sigma_P^E$",
        color="tab:purple",
        linewidth=1.2,
    )
    axes[3].plot(
        time,
        result.sigma_a,
        label=r"$\sigma_A$",
        color="tab:orange",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[3].plot(
        time,
        result.sigma_phi,
        label=r"$\sigma_\Phi$",
        color="tab:green",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[3].plot(
        time, result.tci, label="CI", color="0.55", linewidth=1.0, linestyle="--"
    )
    axes[3].plot(
        time,
        result.tcie,
        label="CI^E",
        color="tab:red",
        linewidth=1.4,
    )
    axes[3].set_ylim(0.0, 1.05)
    axes[3].set_ylabel("CI")
    axes[3].set_xlabel("Time step")
    axes[3].legend(loc="lower left", ncol=6)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_particle_tracking_ablation_figure(
    results: dict[str, TPTResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    colors = {
        "full": "tab:blue",
        "fm1": "tab:purple",
        "fm2": "tab:orange",
        "fm3": "tab:red",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for condition in ABLATION_CONDITIONS:
        result = results[condition]
        axes[0].plot(
            time,
            rolling_mean(np.abs(result.tracking_error), 20),
            color=colors[condition],
            linewidth=1.4,
            label=_condition_label(condition),
        )
        axes[1].plot(
            time,
            rolling_mean(result.tcie, 20),
            color=colors[condition],
            linewidth=1.4,
            label=_condition_label(condition),
        )

    axes[0].set_title(
        "Particle-tracker ablation: rolling absolute tracking error and CI"
    )
    axes[0].set_ylabel("|tracking error|")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].set_ylabel("CI")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower left", ncol=2)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_coercive_masking_figure(
    results: dict[str, TPTResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(next(iter(results.values())).config.steps)
    passive = results["passive"]
    coercive = results["coercive"]
    passive_summary = summarize_result(passive)
    coercive_summary = summarize_result(coercive)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    regime_specs = [
        (
            "Passive FM-1",
            passive,
            passive_summary,
            "tab:green",
            "tab:blue",
            "tab:gray",
        ),
        (
            "Coercive FM-1",
            coercive,
            coercive_summary,
            "tab:purple",
            "tab:orange",
            "tab:red",
        ),
    ]

    for col, (
        title,
        result,
        summary,
        error_color,
        effort_color,
        score_color,
    ) in enumerate(regime_specs):
        axes[0, col].plot(
            time,
            rolling_mean(np.abs(result.tracking_error), 20),
            color=error_color,
            linewidth=1.5,
        )
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("|tracking error|")

        axes[1, col].plot(
            time,
            rolling_mean(result.action_gap, 20),
            color=error_color,
            linewidth=1.4,
            label="action gap",
        )
        axes[1, col].plot(
            time,
            rolling_mean(result.effort_signal, 20),
            color=effort_color,
            linewidth=1.4,
            linestyle="--",
            label="effort",
        )
        axes[1, col].set_ylabel("Effort")
        axes[1, col].legend(loc="upper left")

        if title.startswith("Passive"):
            axes[2, col].plot(
                time,
                rolling_mean(result.tci, 20),
                color=score_color,
                linewidth=1.4,
                label="score",
            )
            axes[2, col].text(
                0.02,
                0.96,
                (
                    f"score={summary['mean_tci']:.3f}\n"
                    f"effort-corrected={summary['mean_tcie']:.3f}\n"
                    f"effort={summary['mean_effort']:.3f}"
                ),
                transform=axes[2, col].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )
            axes[2, col].text(
                0.60,
                0.17,
                "TCI = TCIE",
                transform=axes[2, col].transAxes,
                fontsize=9,
                color="0.35",
            )
        else:
            axes[2, col].plot(
                time,
                rolling_mean(result.tci, 20),
                color="tab:gray",
                linewidth=1.2,
                linestyle="--",
                label="score",
            )
            axes[2, col].plot(
                time,
                rolling_mean(result.tcie, 20),
                color=score_color,
                linewidth=1.4,
                label="effort-corrected score",
            )
            axes[2, col].text(
                0.02,
                0.96,
                (
                    f"score={summary['mean_tci']:.3f}\n"
                    f"effort-corrected={summary['mean_tcie']:.3f}\n"
                    f"effort={summary['mean_effort']:.3f}"
                ),
                transform=axes[2, col].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            )
            axes[2, col].legend(loc="lower left")

        axes[2, col].set_ylabel("CI")
        axes[2, col].set_xlabel("Time step")
        axes[2, col].set_ylim(0.0, 1.05)

    fig.suptitle("Coercive masking: CI stays high while CI^E pays for effort")
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)
