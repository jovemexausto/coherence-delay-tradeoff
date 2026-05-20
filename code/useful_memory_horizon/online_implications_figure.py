from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .online_horizon_adaptation import (
    OnlineAdaptationConfig,
    run_online_horizon_adaptation_experiment,
)
from .twonn_geometry import TwonnGeometryConfig, run_twonn_geometry_experiment


def save_online_implications_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    twonn = run_twonn_geometry_experiment(
        TwonnGeometryConfig(
            holder_exponents=(0.5, 0.75),
            time_steps=300,
            sample_size_per_time=256,
            path_seed_count=4,
            history=220,
        )
    )
    online = run_online_horizon_adaptation_experiment(OnlineAdaptationConfig(seed=7))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    h_labels = [f"H={row.holder_exponent:.2f}" for row in twonn.rows]
    x = np.arange(len(h_labels), dtype=float)
    width = 0.34
    axes[0].bar(
        x - width / 2.0,
        [row.naive_holder_mae for row in twonn.rows],
        width=width,
        label="Endpoint slope",
        color="#d95f5f",
    )
    axes[0].bar(
        x + width / 2.0,
        [row.aggregated_holder_mae for row in twonn.rows],
        width=width,
        label="Aggregated lags",
        color="#4f8bc9",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(h_labels)
    axes[0].set_ylabel(r"Mean absolute error of $\widehat H$")
    axes[0].set_title(r"Roughness estimation with $\widehat k\approx 1$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2, linewidth=0.5, axis="y")

    controller_names = [
        "oracle",
        "plugin",
        "activity",
        "structural",
        "adaptive",
        "static",
    ]
    controller_errors = [
        online.mean_oracle_error,
        online.mean_plugin_error,
        online.mean_activity_error,
        online.mean_structural_error,
        online.mean_adaptive_error,
        online.mean_best_static_error,
    ]
    controller_colors = [
        "#444444",
        "#c44e52",
        "#8172b3",
        "#55a868",
        "#4c72b0",
        "#8c8c8c",
    ]
    axes[1].bar(controller_names, controller_errors, color=controller_colors)
    axes[1].set_ylabel("Mean absolute tracking error")
    axes[1].set_title("Online controller comparison")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(alpha=0.2, linewidth=0.5, axis="y")

    axes[2].plot(
        online.time,
        online.oracle_window,
        linewidth=1.4,
        label="oracle",
        color="#222222",
    )
    axes[2].plot(
        online.time,
        online.structural_window,
        linewidth=1.3,
        label="structural",
        color="#55a868",
    )
    axes[2].plot(
        online.time,
        online.adaptive_window,
        linewidth=1.3,
        label="adaptive",
        color="#4c72b0",
    )
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Window")
    axes[2].set_title("Window trajectories")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the online implications figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/figures/online/fig_online_implications.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_online_implications_figure(args.output)


if __name__ == "__main__":
    main()
