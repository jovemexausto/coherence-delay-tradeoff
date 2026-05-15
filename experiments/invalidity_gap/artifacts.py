from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .model import InvalidityGapResult, rolling_error


def save_invalidity_gap_figure(result: InvalidityGapResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = result.representative
    rolling_operating, rolling_oracle = rolling_error(
        trace, result.config.rolling_window
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    useful_line = axes[0].plot(
        trace.time,
        trace.oracle_horizon,
        color="black",
        linewidth=2.0,
        label=r"useful-memory horizon $n_t^*$",
    )[0]
    operating_line = axes[0].axhline(
        result.config.operating_window,
        color="tab:red",
        linewidth=1.5,
        linestyle="--",
        label="operating horizon",
    )
    drift_axis = axes[0].twinx()
    drift_line = drift_axis.plot(
        trace.time,
        trace.drift_path,
        color="tab:purple",
        linewidth=1.6,
        alpha=0.9,
        label=r"local drift $\zeta_t$",
    )[0]
    drift_axis.set_ylabel(r"$\zeta_t$")

    valid_line = None
    if trace.t_valid is not None:
        valid_line = axes[0].axvline(
            trace.t_valid,
            color="tab:red",
            linewidth=1.8,
            linestyle=":",
            zorder=5,
            label=r"$t_{\mathrm{valid}}$",
        )
    detect_line = None
    if trace.t_detect is not None:
        detect_line = axes[0].axvline(
            trace.t_detect,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            zorder=5,
            label=r"$t_{\mathrm{detect}}$",
        )
    if (
        trace.t_valid is not None
        and trace.t_detect is not None
        and trace.t_detect >= trace.t_valid
    ):
        axes[0].axvspan(
            trace.t_valid,
            trace.t_detect,
            color="gold",
            alpha=0.18,
            linewidth=0,
        )

    axes[0].set_ylabel("Horizon")
    axes[0].set_title("Detector-silent staleness and the invalidity gap")
    legend_handles = [useful_line, operating_line, drift_line]
    axes[0].legend(handles=legend_handles, loc="upper right", frameon=False)

    axes[1].plot(
        trace.time,
        rolling_operating,
        color="tab:red",
        linewidth=1.5,
        label="long-horizon error",
    )
    axes[1].plot(
        trace.time,
        rolling_oracle,
        color="black",
        linewidth=1.5,
        label="oracle-horizon error",
    )
    if trace.t_valid is not None:
        axes[1].axvline(
            trace.t_valid,
            color="tab:red",
            linewidth=1.8,
            linestyle=":",
            label=r"$t_{\mathrm{valid}}$",
        )
    if trace.t_detect is not None:
        axes[1].axvline(
            trace.t_detect,
            color="tab:blue",
            linewidth=1.8,
            linestyle="--",
            label=r"$t_{\mathrm{detect}}$",
        )
    if (
        trace.t_valid is not None
        and trace.t_detect is not None
        and trace.t_detect >= trace.t_valid
    ):
        axes[1].axvspan(
            trace.t_valid,
            trace.t_detect,
            color="gold",
            alpha=0.18,
            linewidth=0,
        )
        axes[1].text(
            trace.t_valid + 20,
            float(max(rolling_operating.max(), rolling_oracle.max())) * 0.92,
            rf"$\Delta_{{inv}}={trace.t_detect - trace.t_valid}$",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Rolling MAE")
    axes[1].legend(loc="upper left", frameon=False)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
