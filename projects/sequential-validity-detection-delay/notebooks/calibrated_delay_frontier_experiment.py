# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from useful_memory_horizon.invalidity_gap import (
    InvalidityGapConfig,
    build_calibrated_delay_frontier_rows,
    run_calibrated_delay_frontier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
CSV_ROOT = ARTIFACT_ROOT / "csv" / "calibrated_delay_frontier_experiment"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "calibrated_delay_frontier_experiment"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "calibrated_delay_frontier_experiment"
for root in (CSV_ROOT, FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


def _parse_float_list(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_str_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _plot_summary(df: pd.DataFrame) -> None:
    families = sorted(df["detector"].unique())
    inputs = sorted(df["detector_input"].unique())
    fig, axes = plt.subplots(
        len(families), 1, figsize=(9.0, 2.6 * len(families)), sharex=True
    )
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families, strict=False):
        sub = df[df["detector"] == family]
        for det_input in inputs:
            ss = sub[sub["detector_input"] == det_input].sort_values(
                ["holder_exponent", "high_drift"]
            )
            label = det_input.replace("_", " ")
            ax.plot(
                range(len(ss)),
                ss["mean_gap"].to_numpy(),
                marker="o",
                label=label,
            )
        ax.axhline(0.0, color="black", lw=1.0, ls="--")
        ax.set_title(family)
        ax.set_ylabel("mean gap")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("grid cells (ordered)")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "mean_gap_by_detector_family.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot = (
        df.groupby(["detector", "detector_input"])["positive_gap_rate"]
        .mean()
        .reset_index()
    )
    x = range(len(pivot))
    labels = [
        f"{r.detector}\n{r.detector_input}" for r in pivot.itertuples(index=False)
    ]
    ax.bar(x, pivot["positive_gap_rate"], color="#39d0d8", edgecolor="#30363d")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean P(gap > 0)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "positive_gap_by_detector_input.png", dpi=200)
    plt.close(fig)


def main() -> None:
    detector_names = _parse_str_list(
        "DELAY_DETECTORS", ("adwin", "page_hinkley", "kswin", "cusum")
    )
    detector_inputs = _parse_str_list(
        "DELAY_INPUTS", ("observation", "absolute_residual")
    )
    holder_exponents = _parse_float_list("DELAY_HS", (0.30, 0.50, 0.75, 1.00))
    false_alarm_targets = _parse_float_list("DELAY_ALPHAS", (0.05, 0.10, 0.20))
    high_drifts = _parse_float_list("DELAY_ZETAS", (0.002, 0.005, 0.010))
    operating_windows = _parse_int_list("DELAY_WINDOWS", (140, 180, 240))
    candidate_deltas = _parse_float_list("DELAY_DELTAS", (0.0005, 0.001, 0.002, 0.004))
    calibration_seeds = _parse_int_list("DELAY_CALIB_SEEDS", tuple(range(100, 106)))
    base_config = InvalidityGapConfig(
        seeds=tuple(range(10, 16)),
        steps=3600,
        warmup=400,
        phase_lengths=(1000, 1400, 1200),
        low_drift=0.00008,
        high_drift=0.0025,
        operating_window=220,
        persistence=40,
    )

    summaries = run_calibrated_delay_frontier(
        detector_names=detector_names,
        detector_inputs=detector_inputs,
        holder_exponents=holder_exponents,
        false_alarm_targets=false_alarm_targets,
        high_drifts=high_drifts,
        operating_windows=operating_windows,
        candidate_deltas=candidate_deltas,
        calibration_seeds=calibration_seeds,
        base_config=base_config,
    )
    rows = build_calibrated_delay_frontier_rows(summaries)
    df = pd.DataFrame(rows)

    df.to_csv(CSV_ROOT / "calibrated_delay_frontier_rows.csv", index=False)
    summary = {
        "row_count": int(len(df)),
        "detectors": detector_names,
        "inputs": detector_inputs,
        "holder_exponents": holder_exponents,
        "false_alarm_targets": false_alarm_targets,
        "high_drifts": high_drifts,
        "operating_windows": operating_windows,
        "mean_gap_by_detector": df.groupby("detector")["mean_gap"].mean().to_dict(),
        "positive_gap_rate_by_detector": df.groupby("detector")["positive_gap_rate"]
        .mean()
        .to_dict(),
        "mean_gap_by_input": df.groupby("detector_input")["mean_gap"].mean().to_dict(),
        "positive_gap_rate_by_input": df.groupby("detector_input")["positive_gap_rate"]
        .mean()
        .to_dict(),
    }
    (TABLE_ROOT / "calibrated_delay_frontier_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _plot_summary(df)

    print("Calibrated delay frontier experiment completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
