from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.common import rolling_mean
from .model import RajputResult


def save_rajput_figure(result: RajputResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_y = result.targets[result.test_slice]
    methods = {
        "single": result.single_mean[result.test_slice],
        "naive": result.naive_mean[result.test_slice],
        "uq": result.uq_mean[result.test_slice],
        "uq+umr": result.uq_umr_mean[result.test_slice],
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for label, pred in methods.items():
        axes[0].plot(
            rolling_mean(np.abs(test_y - pred), 50),
            label=label,
            linewidth=1.4,
        )
    axes[0].set_title(f"{result.config.dataset.upper()} Rajput-style ensemble")
    axes[0].set_ylabel("Rolling MAE")
    axes[0].legend(frameon=False, ncol=2)

    test_cap = result.caps[result.test_slice]
    axes[1].plot(test_cap, color="tab:purple", linewidth=1.2, label="UMR cap")
    axes[1].axhline(
        float(result.buffer_sizes[result.single_index]),
        color="tab:gray",
        linestyle="--",
        linewidth=1.1,
        label="best single buffer",
    )
    axes[1].set_ylabel("Horizon")
    axes[1].set_xlabel("Test step")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
