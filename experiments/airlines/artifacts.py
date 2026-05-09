from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import AirlinesBenchmarkResult


def save_airlines_figure(result: AirlinesBenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(result.targets.size)
    test_mask = np.zeros(result.targets.size, dtype=bool)
    test_mask[result.test_slice] = True

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    accuracy_series = {
        "single": (result.single_probabilities >= 0.5).astype(float) == result.targets,
        "ensemble": (result.ensemble_probabilities >= 0.5).astype(float)
        == result.targets,
        "single + cap": (result.single_cap_probabilities >= 0.5).astype(float)
        == result.targets,
        "ensemble + cap": (result.ensemble_cap_probabilities >= 0.5).astype(float)
        == result.targets,
    }
    for label, series in accuracy_series.items():
        axis = axes[0]
        axis.plot(
            time, rolling_mean(series.astype(float), 250), label=label, linewidth=1.2
        )

    axes[0].set_ylabel("Rolling accuracy")
    axes[0].set_title("Airlines prequential classification cost of horizon caps")
    axes[0].legend(loc="lower right", ncol=2)

    axes[1].plot(time, result.caps, color="tab:purple", linewidth=1.2, label="cap")
    axes[1].plot(
        time,
        result.drift_proxy,
        color="tab:orange",
        linewidth=0.9,
        alpha=0.75,
        label="drift proxy",
    )
    axes[1].axhline(
        result.selected_window,
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
        label="selected window",
    )
    axes[1].set_ylabel("Window")
    axes[1].set_xlabel("Time step")
    axes[1].legend(loc="upper right", ncol=3)

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
        axis.axvspan(
            result.test_slice.start, result.test_slice.stop, color="0.9", alpha=0.35
        )

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(fig)
