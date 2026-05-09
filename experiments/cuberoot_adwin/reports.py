from __future__ import annotations

import numpy as np

from ..core.types import SummaryRows
from .model import CubeRootADWINBenchmarkResult, _fixed_window_estimate


def build_summary_rows(result: CubeRootADWINBenchmarkResult) -> SummaryRows:
    rows: list[dict[str, float | str | int]] = []
    for method, summary in result.summaries.items():
        rows.append(
            {
                "method": method,
                "tail_mae_mean": round(summary.tail_mae_mean, 4),
                "tail_mae_std": round(summary.tail_mae_std, 4),
                "tail_rmse_mean": round(summary.tail_rmse_mean, 4),
                "tail_rmse_std": round(summary.tail_rmse_std, 4),
                "tail_width_mean": round(summary.tail_width_mean, 2),
                "tail_width_std": round(summary.tail_width_std, 2),
                "event_count_mean": round(summary.event_count_mean, 2),
                "event_count_std": round(summary.event_count_std, 2),
                "cap_count_mean": ""
                if summary.cap_count_mean is None
                else round(summary.cap_count_mean, 2),
                "cap_only_count_mean": ""
                if summary.cap_only_count_mean is None
                else round(summary.cap_only_count_mean, 2),
                "first_cap_time_mean": ""
                if summary.first_cap_time_mean is None
                else round(summary.first_cap_time_mean, 1),
                "first_drift_time_mean": ""
                if summary.first_drift_time_mean is None
                else round(summary.first_drift_time_mean, 1),
                "cap_before_drift_delay_mean": ""
                if summary.cap_before_drift_delay_mean is None
                else round(summary.cap_before_drift_delay_mean, 1),
            }
        )
    return rows


def build_event_rows(result: CubeRootADWINBenchmarkResult) -> SummaryRows:
    representative = result.representative
    return [
        {
            "event": "drift_detected",
            "method": "adwin",
            "count": len(representative.adwin_drift_detected),
            "first_time": representative.adwin_drift_detected[0]
            if representative.adwin_drift_detected
            else "",
        },
        {
            "event": "drift_detected",
            "method": "cube",
            "count": len(representative.cube_drift_detected),
            "first_time": representative.cube_drift_detected[0]
            if representative.cube_drift_detected
            else "",
        },
        {
            "event": "cap_triggered",
            "method": "cube",
            "count": len(representative.cube_cap_triggered),
            "first_time": representative.cube_cap_triggered[0]
            if representative.cube_cap_triggered
            else "",
        },
        {
            "event": "cap_only",
            "method": "cube",
            "count": len(
                set(representative.cube_cap_triggered)
                - set(representative.cube_drift_detected)
            ),
            "first_time": next(
                (
                    t
                    for t in representative.cube_cap_triggered
                    if t not in set(representative.cube_drift_detected)
                ),
                "",
            ),
        },
    ]


def build_phase_rows(result: CubeRootADWINBenchmarkResult) -> SummaryRows:
    if result.config.piecewise_drifts:
        drifts = result.config.piecewise_drifts
        edges = np.linspace(0, result.config.steps, len(drifts) + 1, dtype=int)
    else:
        drifts = (result.config.drift,)
        edges = np.asarray([0, result.config.steps], dtype=int)

    rows: list[dict[str, float | int | str]] = []
    for phase, drift in enumerate(drifts):
        start = int(edges[phase])
        stop = int(edges[phase + 1])
        for method, series in result.series.items():
            rows.append(
                {
                    "phase": phase,
                    "start": start,
                    "stop": stop,
                    "drift": float(drift),
                    "method": method,
                    "mean_abs_error": round(
                        float(np.mean(series.absolute_error_mean[start:stop])),
                        4,
                    ),
                    "mean_width": round(
                        float(np.mean(series.memory_horizon_mean[start:stop])),
                        2,
                    ),
                }
            )
    return rows


def build_oracle_phase_rows(result: CubeRootADWINBenchmarkResult) -> SummaryRows:
    if result.config.piecewise_drifts:
        drifts = result.config.piecewise_drifts
        edges = np.linspace(0, result.config.steps, len(drifts) + 1, dtype=int)
    else:
        drifts = (result.config.drift,)
        edges = np.asarray([0, result.config.steps], dtype=int)

    rows: list[dict[str, float | int | str]] = []
    for phase, drift in enumerate(drifts):
        start = int(edges[phase])
        stop = int(edges[phase + 1])
        oracle_scores: list[tuple[int, float]] = []
        for window in result.config.oracle_windows:
            per_seed_errors: list[float] = []
            for trace in result.traces:
                estimate, _ = _fixed_window_estimate(trace.observations, window)
                per_seed_errors.append(
                    float(
                        np.mean(
                            np.abs(trace.latent_mean[start:stop] - estimate[start:stop])
                        )
                    )
                )
            oracle_scores.append((window, float(np.mean(per_seed_errors))))
        oracle_window, oracle_error = min(oracle_scores, key=lambda item: item[1])

        cube = result.series["cube"]
        fixed = result.series["fixed"]
        fixed_long = result.series["fixed_long"]
        ewma = result.series["ewma"]
        adwin = result.series["adwin"]
        rows.append(
            {
                "phase": phase,
                "start": start,
                "stop": stop,
                "drift": float(drift),
                "oracle_window": int(oracle_window),
                "oracle_mae": round(oracle_error, 4),
                "cube_mae": round(
                    float(np.mean(cube.absolute_error_mean[start:stop])), 4
                ),
                "cube_mean_n_star": round(
                    float(np.mean(cube.memory_horizon_mean[start:stop])), 2
                ),
                "fixed_mae": round(
                    float(np.mean(fixed.absolute_error_mean[start:stop])), 4
                ),
                "fixed_long_mae": round(
                    float(np.mean(fixed_long.absolute_error_mean[start:stop])), 4
                ),
                "ewma_mae": round(
                    float(np.mean(ewma.absolute_error_mean[start:stop])), 4
                ),
                "adwin_mae": round(
                    float(np.mean(adwin.absolute_error_mean[start:stop])), 4
                ),
                "adwin_mean_width": round(
                    float(np.mean(adwin.memory_horizon_mean[start:stop])), 2
                ),
            }
        )
    return rows


def build_frontier_rows(
    result: CubeRootADWINBenchmarkResult,
    windows: tuple[int, ...] | None = None,
) -> SummaryRows:
    windows = windows or tuple(range(10, 501, 10))
    rows: list[dict[str, float | int | str]] = []

    for window in windows:
        per_seed_mae: list[float] = []
        per_seed_width: list[float] = []
        for trace in result.traces:
            estimate, width = _fixed_window_estimate(trace.observations, window)
            tail_start = int(estimate.size * (1.0 - result.config.tail_fraction))
            per_seed_mae.append(
                float(
                    np.mean(
                        np.abs(trace.latent_mean[tail_start:] - estimate[tail_start:])
                    )
                )
            )
            per_seed_width.append(float(np.mean(width[tail_start:])))

        rows.append(
            {
                "method": "fixed_sweep",
                "window": int(window),
                "tail_mae_mean": round(float(np.mean(per_seed_mae)), 4),
                "tail_mae_std": round(float(np.std(per_seed_mae)), 4),
                "tail_width_mean": round(float(np.mean(per_seed_width)), 2),
                "tail_width_std": round(float(np.std(per_seed_width)), 2),
            }
        )

    for method in ("fixed", "fixed_long", "ewma", "adwin", "cube"):
        summary = result.summaries[method]
        rows.append(
            {
                "method": method,
                "window": "",
                "tail_mae_mean": round(summary.tail_mae_mean, 4),
                "tail_mae_std": round(summary.tail_mae_std, 4),
                "tail_width_mean": round(summary.tail_width_mean, 2),
                "tail_width_std": round(summary.tail_width_std, 2),
            }
        )

    return rows


def build_delay_rows(result: CubeRootADWINBenchmarkResult) -> SummaryRows:
    cube = result.summaries["cube"]
    adwin = result.summaries["adwin"]
    lead_time = (
        None
        if cube.first_cap_time_mean is None or adwin.first_drift_time_mean is None
        else float(adwin.first_drift_time_mean - cube.first_cap_time_mean)
    )
    return [
        {
            "drift": round(float(result.config.drift), 6),
            "cube_first_cap_time_mean": ""
            if cube.first_cap_time_mean is None
            else round(cube.first_cap_time_mean, 2),
            "adwin_first_drift_time_mean": ""
            if adwin.first_drift_time_mean is None
            else round(adwin.first_drift_time_mean, 2),
            "lead_time_mean": "" if lead_time is None else round(lead_time, 2),
            "cube_cap_only_count_mean": ""
            if cube.cap_only_count_mean is None
            else round(cube.cap_only_count_mean, 2),
        }
    ]
