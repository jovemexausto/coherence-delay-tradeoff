from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import CubeRootADWINBenchmarkResult
from .reports import build_delay_rows, build_frontier_rows


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


def save_frontier_figure(
    result: CubeRootADWINBenchmarkResult, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_frontier_rows(result)
    sweep_rows = [row for row in rows if row["method"] == "fixed_sweep"]
    point_rows = [row for row in rows if row["method"] != "fixed_sweep"]

    sweep_x = np.asarray([float(row["tail_width_mean"]) for row in sweep_rows])
    sweep_y = np.asarray([float(row["tail_mae_mean"]) for row in sweep_rows])
    order = np.argsort(sweep_x)
    sweep_x = sweep_x[order]
    sweep_y = sweep_y[order]

    best_index = int(np.argmin(sweep_y))
    best_x = float(sweep_x[best_index])
    best_y = float(sweep_y[best_index])

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.plot(
        sweep_x,
        sweep_y,
        color="black",
        linewidth=2.0,
        label="fixed-window sweep",
    )
    ax.scatter(
        [best_x],
        [best_y],
        s=70,
        color="black",
        zorder=4,
        label="oracle fixed window",
    )

    xmin = float(np.min(sweep_x))
    xmax = float(np.max(sweep_x))
    ax.axvspan(xmin, best_x * 0.9, color="tab:blue", alpha=0.05, linewidth=0)
    ax.axvspan(best_x * 0.9, best_x * 1.1, color="gold", alpha=0.08, linewidth=0)
    ax.axvspan(best_x * 1.1, xmax, color="tab:red", alpha=0.05, linewidth=0)

    palette = {
        "fixed": "tab:gray",
        "fixed_long": "tab:brown",
        "ewma": "tab:blue",
        "adwin": "tab:orange",
        "cube": "tab:green",
    }
    labels = {
        "fixed": "fixed-100",
        "fixed_long": "fixed-500",
        "ewma": "EWMA",
        "adwin": "ADWIN",
        "cube": "CubeRootADWIN",
    }
    for row in point_rows:
        method = str(row["method"])
        x = float(row["tail_width_mean"])
        y = float(row["tail_mae_mean"])
        ax.scatter(
            [x],
            [y],
            s=85 if method == "cube" else 65,
            color=palette[method],
            zorder=5,
            label=labels[method],
        )

    ax.axvline(best_x, color="black", linestyle="--", linewidth=1.1, alpha=0.6)

    ax.set_xlabel("Effective memory horizon")
    ax.set_ylabel("Tail MAE")
    ax.set_title("Lag-variance frontier for useful memory")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_delay_figure(
    results: list[CubeRootADWINBenchmarkResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        rows.extend(dict(row) for row in build_delay_rows(result))

    drifts = np.asarray([float(row["drift"]) for row in rows])
    cap_times = np.asarray([float(row["cube_first_cap_time_mean"]) for row in rows])
    adwin_times = np.asarray(
        [float(row["adwin_first_drift_time_mean"]) for row in rows]
    )
    lead_times = np.asarray([float(row["lead_time_mean"]) for row in rows])

    order = np.argsort(drifts)
    drifts = drifts[order]
    cap_times = cap_times[order]
    adwin_times = adwin_times[order]
    lead_times = lead_times[order]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)

    axes[0].plot(
        drifts,
        cap_times,
        color="tab:green",
        linewidth=2.0,
        marker="o",
        label="CubeRootADWIN cap",
    )
    axes[0].plot(
        drifts,
        adwin_times,
        color="tab:orange",
        linewidth=2.0,
        marker="o",
        label="ADWIN detection",
    )
    axes[0].fill_between(
        drifts,
        cap_times,
        adwin_times,
        color="gold",
        alpha=0.14,
        label="temporal validity gap",
    )
    axes[0].set_ylabel("First event time")
    axes[0].set_title("Cap-before-detection gap under continuous drift")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(alpha=0.2, linewidth=0.5)
    axes[0].set_xscale("log")

    axes[1].plot(
        drifts,
        lead_times,
        color="black",
        linewidth=2.0,
        marker="o",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel("Drift rate")
    axes[1].set_ylabel("Lead time")
    axes[1].grid(alpha=0.2, linewidth=0.5)
    axes[1].set_xscale("log")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
