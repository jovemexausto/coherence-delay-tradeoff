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

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))

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
    axes[0].set_title(r"Aggregated lags stabilize roughness estimation")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2, linewidth=0.5, axis="y")
    axes[0].text(
        0.03,
        0.97,
        r"median $\widehat{k}\approx 1$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    controller_names = ["plugin", "activity", "structural", "adaptive", "static"]
    controller_errors = [
        online.mean_plugin_error,
        online.mean_activity_error,
        online.mean_structural_error,
        online.mean_adaptive_error,
        online.mean_best_static_error,
    ]
    controller_excess_pct = [
        100.0 * (error / online.mean_oracle_error - 1.0) for error in controller_errors
    ]
    controller_colors = [
        "#c44e52",
        "#8172b3",
        "#55a868",
        "#4c72b0",
        "#8c8c8c",
    ]
    axes[1].bar(controller_names, controller_excess_pct, color=controller_colors)
    axes[1].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1].set_ylabel("Excess tracking error over oracle (%)")
    axes[1].set_title("Controller gap relative to oracle")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(alpha=0.2, linewidth=0.5, axis="y")

    phase_starts = np.cumsum((0,) + online.config.phase_lengths[:-1])
    phase_ends = np.cumsum(online.config.phase_lengths)
    phase_labels = [
        rf"phase {idx + 1}" + "\n" + rf"$H={H:.2f}$"
        for idx, H in enumerate(online.config.holder_exponents)
    ]
    oracle_phase_means = []
    structural_phase_means = []
    adaptive_phase_means = []
    for start, end in zip(phase_starts, phase_ends, strict=True):
        oracle_phase_means.append(float(np.mean(online.oracle_window[start:end])))
        structural_phase_means.append(
            float(np.mean(online.structural_window[start:end]))
        )
        adaptive_phase_means.append(float(np.mean(online.adaptive_window[start:end])))

    phase_x = np.arange(len(phase_labels), dtype=float)
    phase_width = 0.24
    axes[2].bar(
        phase_x - phase_width,
        oracle_phase_means,
        width=phase_width,
        color="#222222",
        label="oracle",
    )
    axes[2].bar(
        phase_x,
        structural_phase_means,
        width=phase_width,
        color="#55a868",
        label="structural",
    )
    axes[2].bar(
        phase_x + phase_width,
        adaptive_phase_means,
        width=phase_width,
        color="#4c72b0",
        label="adaptive",
    )
    axes[2].set_xticks(phase_x)
    axes[2].set_xticklabels(phase_labels)
    axes[2].set_ylabel("Mean window within phase")
    axes[2].set_title("Window scale by drift phase")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.2, linewidth=0.5, axis="y")

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
