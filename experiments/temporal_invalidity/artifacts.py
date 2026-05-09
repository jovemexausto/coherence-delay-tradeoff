from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import TemporalInvalidityResult


def _shade_segments(
    axis: plt.Axes, segments: list[tuple[int, int]], *, color: str
) -> None:
    for start, end in segments:
        axis.axvspan(start, end, color=color, alpha=0.18, linewidth=0)


def save_temporal_invalidity_figure(
    result: TemporalInvalidityResult, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace = result.representative
    time = result.time

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, trace.true_zeta, color="tab:purple", linewidth=1.4)
    axes[0].set_ylabel(r"$\zeta_{true}(t)$")
    axes[0].set_title("Downstream cost of temporal invalidity")
    for start, end in ((0, 2000), (2000, 4000), (4000, 6000)):
        axes[0].axvspan(start, end, color="0.9", alpha=0.08)
    axes[0].text(950, float(np.max(trace.true_zeta)) * 0.95, "Phase 1", ha="center")
    axes[0].text(3000, float(np.max(trace.true_zeta)) * 0.95, "Phase 2", ha="center")
    axes[0].text(5000, float(np.max(trace.true_zeta)) * 0.95, "Phase 3", ha="center")

    axes[1].plot(
        time, trace.umr_width_policy, label="UMR", color="tab:red", linewidth=1.2
    )
    axes[1].plot(
        time, trace.adwin_width_policy, label="ADWIN", color="tab:blue", linewidth=1.0
    )
    axes[1].axhline(400, color="0.3", linestyle="--", linewidth=1.0, label="Fixed-400")
    axes[1].axhline(100, color="0.5", linestyle=":", linewidth=1.0, label="Fixed-100")
    axes[1].set_ylabel("Effective horizon")
    axes[1].legend(loc="upper right", ncol=2)
    _shade_segments(axes[1], result.cap_only_segments, color="tab:orange")
    for event in np.flatnonzero(trace.adwin_event):
        axes[1].axvline(event, color="tab:blue", alpha=0.12, linewidth=0.8)

    rolling = {
        "Fixed-400": rolling_mean(
            trace.fixed_400_accuracy, result.config.rolling_window
        ),
        "Fixed-100": rolling_mean(
            trace.fixed_100_accuracy, result.config.rolling_window
        ),
        "ADWIN": rolling_mean(trace.adwin_accuracy, result.config.rolling_window),
        "UMR": rolling_mean(trace.umr_accuracy, result.config.rolling_window),
    }
    for label, series in rolling.items():
        axes[2].plot(time, series, label=label, linewidth=1.2)
    _shade_segments(axes[2], result.cap_only_segments, color="tab:orange")
    axes[2].set_ylabel("Rolling accuracy")
    axes[2].set_xlabel("Time step")
    axes[2].legend(loc="lower right", ncol=2)

    if result.cap_only_segments:
        cap_only_mask = trace.cap_only_mask
        start, end = np.flatnonzero(cap_only_mask)[[0, -1]]
        delta = 100.0 * (
            np.nanmean(trace.fixed_400_accuracy[cap_only_mask])
            - np.nanmean(trace.umr_accuracy[cap_only_mask])
        )
        axes[2].text(
            start + 20,
            float(np.nanmax(list(rolling.values()))) - 0.02,
            f"cap-only: {delta:.1f} pp",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9),
        )

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
