from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..core.common import rolling_mean
from .model import CubeRootADWINBenchmarkResult


def save_benchmark_figure(
    result: CubeRootADWINBenchmarkResult, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rep = result.representative
    time = result.time

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(
        time, rolling_mean(rep.fixed_error, 50), label="fixed-100", linewidth=1.3
    )
    axes[0].plot(
        time,
        rolling_mean(rep.fixed_long_error, 50),
        label="fixed-500",
        linewidth=1.3,
    )
    axes[0].plot(time, rolling_mean(rep.ewma_error, 50), label="EWMA", linewidth=1.3)
    axes[0].plot(time, rolling_mean(rep.adwin_error, 50), label="ADWIN", linewidth=1.3)
    axes[0].plot(
        time, rolling_mean(rep.cube_error, 50), label="CubeRootADWIN", linewidth=1.3
    )
    axes[0].set_ylabel("Rolling MAE")
    axes[0].set_title("Continuous drift tracking")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(time, rep.fixed_width, label="fixed-100", linewidth=1.2)
    axes[1].plot(time, rep.fixed_long_width, label="fixed-500", linewidth=1.2)
    axes[1].plot(time, rep.ewma_width, label="EWMA effective width", linewidth=1.2)
    axes[1].plot(time, rep.adwin_width, label="ADWIN width", linewidth=1.2)
    axes[1].plot(time, rep.cube_width, label="CubeRootADWIN width", linewidth=1.2)
    axes[1].plot(
        time,
        rep.cube_n_star,
        label=r"CubeRootADWIN $n^*$",
        linewidth=1.1,
        linestyle="--",
    )
    axes[1].set_ylabel("Memory horizon")
    axes[1].legend(loc="upper left", ncol=3)

    for t in rep.adwin_drift_detected:
        axes[2].axvline(t, color="tab:blue", alpha=0.35, linewidth=0.8)
    for t in rep.cube_drift_detected:
        axes[2].axvline(t, color="tab:orange", alpha=0.35, linewidth=0.8)
    for t in rep.cube_cap_triggered:
        axes[2].axvline(t, color="tab:green", alpha=0.45, linewidth=1.0)
    axes[2].set_ylabel("Events")
    axes[2].set_xlabel("Time step")
    axes[2].set_yticks([])
    axes[2].text(0.01, 0.75, "blue=ADWIN drift", transform=axes[2].transAxes)
    axes[2].text(0.01, 0.55, "orange=CubeRoot drift", transform=axes[2].transAxes)
    axes[2].text(0.01, 0.35, "green=cube cap", transform=axes[2].transAxes)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
