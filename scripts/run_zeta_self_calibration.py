from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cuberoot_adwin.model import UMRBenchmarkConfig, run_benchmark


CSV_PATH = ROOT / "artifacts" / "cuberoot_adwin" / "zeta_self_calibration.csv"
FIG_PATH = ROOT / "figures" / "cuberoot_adwin" / "fig_zeta_self_calibration.pdf"


def _base_configs() -> dict[str, UMRBenchmarkConfig]:
    return {
        "constant": UMRBenchmarkConfig(
            seeds=tuple(range(8)),
            drift=0.001,
            fixed_window=100,
            fixed_long_window=500,
            ewma_alpha=0.05,
            adwin_delta=0.002,
            drift_window=100,
            drift_ema_alpha=0.05,
            n_min=10,
            n_max=500,
            tail_fraction=0.5,
            calibration_prefix=500,
        ),
        "alternating": UMRBenchmarkConfig(
            seeds=tuple(range(8)),
            piecewise_drifts=(0.0002, 0.0045, 0.00015, 0.007, 0.00025),
            piecewise_lengths=(900, 140, 700, 70, 1190),
            fixed_window=100,
            fixed_long_window=500,
            ewma_alpha=0.05,
            adwin_delta=0.002,
            drift_window=100,
            drift_ema_alpha=0.05,
            n_min=10,
            n_max=500,
            tail_fraction=0.5,
            calibration_prefix=500,
        ),
    }


def _observable_prefix_score(result, prefix_steps: int) -> float:
    trace = result.representative
    estimate = np.asarray(trace.cube_estimate[:prefix_steps], dtype=float)
    observations = np.asarray(trace.observations[:prefix_steps], dtype=float)
    # Use one-step-ahead forecast error on the observable stream.
    valid = np.isfinite(estimate[:-1])
    if not np.any(valid):
        return float("inf")
    forecast = estimate[:-1][valid]
    target = observations[1:][valid]
    return float(np.mean((target - forecast) ** 2))


def _evaluate_one(
    label: str,
    config: UMRBenchmarkConfig,
    seed: int,
    candidate_windows: tuple[int, ...],
    candidate_alphas: tuple[float, ...],
) -> dict[str, float | int | str]:
    prefix_steps = config.calibration_prefix
    best_candidate: tuple[int, float, float] | None = None
    for window in candidate_windows:
        for alpha in candidate_alphas:
            prefix_config = replace(
                config,
                seeds=(seed,),
                steps=prefix_steps,
                drift_window=window,
                drift_ema_alpha=alpha,
                tail_fraction=0.5,
            )
            prefix_result = run_benchmark(prefix_config)
            score = _observable_prefix_score(prefix_result, prefix_steps)
            candidate = (window, alpha, score)
            if best_candidate is None or score < best_candidate[2]:
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No calibration candidate selected")
    window, alpha, score = best_candidate

    selected_config = replace(
        config,
        seeds=(seed,),
        drift_window=window,
        drift_ema_alpha=alpha,
    )
    default_config = replace(config, seeds=(seed,))

    selected_result = run_benchmark(selected_config)
    default_result = run_benchmark(default_config)

    selected_cube = selected_result.summaries["cube"]
    default_cube = default_result.summaries["cube"]
    return {
        "scenario": label,
        "seed": seed,
        "selected_window": window,
        "selected_alpha": alpha,
        "calibration_score": score,
        "selected_tail_mae": selected_cube.tail_mae_mean,
        "default_tail_mae": default_cube.tail_mae_mean,
        "tail_mae_delta": default_cube.tail_mae_mean - selected_cube.tail_mae_mean,
        "selected_tail_width": selected_cube.tail_width_mean,
        "default_tail_width": default_cube.tail_width_mean,
        "selected_cap_count": selected_cube.cap_count_mean or 0.0,
        "default_cap_count": default_cube.cap_count_mean or 0.0,
        "selected_cap_only": selected_cube.cap_only_count_mean or 0.0,
        "default_cap_only": default_cube.cap_only_count_mean or 0.0,
        "selected_first_cap": selected_cube.first_cap_time_mean or np.nan,
        "default_first_cap": default_cube.first_cap_time_mean or np.nan,
    }


def _summarize(rows: list[dict[str, float | int | str]]) -> None:
    by_scenario: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    for scenario, scenario_rows in by_scenario.items():
        selected = np.array([float(r["selected_tail_mae"]) for r in scenario_rows])
        default = np.array([float(r["default_tail_mae"]) for r in scenario_rows])
        delta = np.array([float(r["tail_mae_delta"]) for r in scenario_rows])
        alphas = np.array([float(r["selected_alpha"]) for r in scenario_rows])
        windows = np.array([float(r["selected_window"]) for r in scenario_rows])
        print(
            {
                "scenario": scenario,
                "mean_selected_alpha": float(np.mean(alphas)),
                "mean_selected_window": float(np.mean(windows)),
                "default_tail_mae": float(np.mean(default)),
                "selected_tail_mae": float(np.mean(selected)),
                "mean_tail_mae_delta": float(np.mean(delta)),
                "p_selected_better": float(np.mean(delta > 0.0)),
            }
        )


def _plot(rows: list[dict[str, float | int | str]]) -> None:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(10, 4.4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios, strict=True):
        scenario_rows = [row for row in rows if str(row["scenario"]) == scenario]
        default = np.array([float(r["default_tail_mae"]) for r in scenario_rows])
        selected = np.array([float(r["selected_tail_mae"]) for r in scenario_rows])
        x = np.arange(len(scenario_rows))
        ax.bar(x - 0.18, default, width=0.36, label="default", color="0.7")
        ax.bar(x + 0.18, selected, width=0.36, label="calibrated", color="tab:blue")
        ax.set_title(scenario.capitalize())
        ax.set_xlabel("seed")
        ax.set_xticks(x, [str(int(r["seed"])) for r in scenario_rows])
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel("Tail MAE")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Self-calibrated drift proxy vs fixed default", fontsize=13)
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)


def main() -> None:
    candidate_windows = (50, 75, 100, 150)
    candidate_alphas = (0.01, 0.02, 0.05, 0.1, 0.2)

    rows: list[dict[str, float | int | str]] = []
    for label, config in _base_configs().items():
        for seed in config.seeds:
            rows.append(
                _evaluate_one(
                    label,
                    config,
                    seed,
                    candidate_windows,
                    candidate_alphas,
                )
            )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _summarize(rows)
    _plot(rows)
    print(f"Saved {CSV_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
