from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cuberoot_adwin.model import UMRBenchmarkConfig, run_benchmark

ROW_END = r"\\"


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def tail_mae(trace, tail_fraction: float) -> float:
    start = int(trace.cube_error.size * (1.0 - tail_fraction))
    return float(np.mean(trace.cube_error[start:]))


def scenario_configs() -> dict[str, UMRBenchmarkConfig]:
    return {
        "Constant drift": UMRBenchmarkConfig(
            seeds=tuple(range(20)),
            drift=0.001,
            fixed_window=100,
            fixed_long_window=500,
            ewma_alpha=0.05,
            adwin_delta=0.002,
            Ck=1.0,
            drift_window=100,
            drift_ema_alpha=0.05,
            calibration_prefix=500,
        ),
        "Alternating drift": UMRBenchmarkConfig(
            seeds=tuple(range(20)),
            piecewise_drifts=(0.0002, 0.0045, 0.00015, 0.007, 0.00025),
            piecewise_lengths=(900, 140, 700, 70, 1190),
            fixed_window=100,
            fixed_long_window=200,
            ewma_alpha=0.05,
            adwin_delta=0.002,
            Ck=1.0,
            drift_window=100,
            drift_ema_alpha=0.05,
            calibration_prefix=500,
        ),
    }


def run_calibration_study() -> tuple[
    list[dict[str, str | float]], list[dict[str, str | float]]
]:
    window_grid = (25, 50, 75, 100)
    alpha_grid = (0.01, 0.02, 0.05, 0.10, 0.20)
    detail_rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float]] = []

    for scenario, base_config in scenario_configs().items():
        default_result = run_benchmark(base_config)
        for seed, trace in zip(base_config.seeds, default_result.traces, strict=True):
            best_window = base_config.drift_window
            best_alpha = base_config.drift_ema_alpha
            best_score = float("inf")
            best_tail_mae = float("inf")
            prefix = base_config.calibration_prefix
            for window in window_grid:
                for alpha in alpha_grid:
                    candidate_config = replace(
                        base_config,
                        seeds=(seed,),
                        drift_window=window,
                        drift_ema_alpha=alpha,
                    )
                    candidate_result = run_benchmark(candidate_config)
                    candidate_trace = candidate_result.traces[0]
                    score = float(
                        np.mean(
                            (
                                candidate_trace.observations[1:prefix]
                                - candidate_trace.cube_estimate[: prefix - 1]
                            )
                            ** 2
                        )
                    )
                    if score < best_score:
                        best_score = score
                        best_window = window
                        best_alpha = alpha
                        best_tail_mae = tail_mae(
                            candidate_trace,
                            candidate_config.tail_fraction,
                        )
            detail_rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "selected_alpha": round(best_alpha, 4),
                    "selected_window": best_window,
                    "prefix_score": round(best_score, 6),
                    "default_tail_mae": round(
                        tail_mae(trace, base_config.tail_fraction), 4
                    ),
                    "calibrated_tail_mae": round(best_tail_mae, 4),
                }
            )

        scenario_rows = [row for row in detail_rows if row["scenario"] == scenario]
        summary_rows.append(
            {
                "scenario": scenario,
                "mean_alpha": round(
                    float(np.mean([row["selected_alpha"] for row in scenario_rows])), 2
                ),
                "mean_window": round(
                    float(np.mean([row["selected_window"] for row in scenario_rows])), 1
                ),
                "tail_mae_default": round(
                    float(np.mean([row["default_tail_mae"] for row in scenario_rows])),
                    4,
                ),
                "tail_mae_calibrated": round(
                    float(
                        np.mean([row["calibrated_tail_mae"] for row in scenario_rows])
                    ),
                    4,
                ),
            }
        )

    return detail_rows, summary_rows


def save_figure(
    detail_rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    scenarios = [row["scenario"] for row in summary_rows]
    default_mae = [float(row["tail_mae_default"]) for row in summary_rows]
    calibrated_mae = [float(row["tail_mae_calibrated"]) for row in summary_rows]
    x = np.arange(len(scenarios))
    width = 0.35
    axes[0].bar(x - width / 2, default_mae, width=width, label="default")
    axes[0].bar(x + width / 2, calibrated_mae, width=width, label="calibrated")
    axes[0].set_xticks(x, scenarios)
    axes[0].set_ylabel("Tail MAE")
    axes[0].set_title("Prefix-validated calibration")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, linewidth=0.5)

    palette = {"Constant drift": "tab:blue", "Alternating drift": "tab:orange"}
    for scenario in palette:
        rows = [row for row in detail_rows if row["scenario"] == scenario]
        axes[1].scatter(
            [float(row["selected_window"]) for row in rows],
            [float(row["selected_alpha"]) for row in rows],
            s=28,
            alpha=0.8,
            label=scenario,
            color=palette[scenario],
        )
    axes[1].set_xlabel("Selected block window")
    axes[1].set_ylabel(r"Selected $\alpha$")
    axes[1].set_title("Chosen drift-proxy settings")
    axes[1].grid(alpha=0.2, linewidth=0.5)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summary_table(rows: list[dict[str, str | float]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Prefix-validated auto-calibration on the synthetic benchmarks. The calibration prefix chooses $(d,\alpha)$ by observable one-step-ahead prequential error, then the selected parameters are run on the full stream.}",
        r"\label{tab:zeta_autocalibration}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scenario & Mean $\alpha$ & Mean $d$ & Tail MAE (default) & Tail MAE (calibrated) "
        + ROW_END,
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['scenario']} & {float(row['mean_alpha']):.2f} & {float(row['mean_window']):.1f} & {float(row['tail_mae_default']):.4f} & {float(row['tail_mae_calibrated']):.4f} "
            + ROW_END
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate zeta self-calibration artifacts."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/calibration"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("artifacts/tables/calibration"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/cuberoot_adwin"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_rows, summary_rows = run_calibration_study()
    write_csv(args.csv_dir / "zeta_self_calibration_detail.csv", detail_rows)
    write_csv(args.csv_dir / "zeta_self_calibration_summary.csv", summary_rows)
    write_text(
        args.tables_dir / "zeta_autocalibration.tex", summary_table(summary_rows)
    )
    save_figure(
        detail_rows, summary_rows, args.figures_dir / "fig_zeta_self_calibration.pdf"
    )


if __name__ == "__main__":
    main()
