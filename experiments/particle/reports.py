from __future__ import annotations

import numpy as np
from typing import cast

from ..core.common import summarize_onset, threshold_crossings
from ..core.types import SummaryRows
from .model import (
    ACTIVE_BASELINE_DETECTORS,
    TPTConfig,
    TPTActiveBenchmarkResult,
    TPTResult,
    run_coercive_masking_experiment,
)


def summarize_result(result: TPTResult, tail_window: int = 60) -> dict[str, float]:
    tail = slice(-min(tail_window, result.config.steps), None)
    return {
        "mean_abs_error": float(np.mean(np.abs(result.tracking_error[tail]))),
        "mean_action_gap": float(np.mean(result.action_gap[tail])),
        "mean_effort": float(np.mean(result.effort_signal[tail])),
        "mean_sigma_p": float(np.mean(result.sigma_p[tail])),
        "mean_sigma_p_eff": float(np.mean(result.sigma_p_eff[tail])),
        "mean_sigma_a": float(np.mean(result.sigma_a[tail])),
        "mean_sigma_phi": float(np.mean(result.sigma_phi[tail])),
        "mean_tci": float(np.mean(result.tci[tail])),
        "mean_tcie": float(np.mean(result.tcie[tail])),
        "resampling_steps": float(np.sum(result.resampled)),
    }


def build_active_benchmark_rows(
    results: list[TPTActiveBenchmarkResult],
) -> SummaryRows:
    rows: list[dict[str, str | float | int]] = []
    for phase in ("masking_detection", "collapse_detection"):
        for detector in ("TCI", "TCIE", *ACTIVE_BASELINE_DETECTORS):
            delays: list[float] = []
            detections = 0
            for result in results:
                summary = getattr(result, phase)[detector]
                delay = summary.delay
                if delay is not None:
                    detections += 1
                    delays.append(float(delay))
            rows.append(
                {
                    "phase": phase.replace("_", " "),
                    "detector": detector,
                    "detections": detections,
                    "n_runs": len(results),
                    "detection_rate": round(detections / len(results), 3),
                    "mean_delay": round(float(np.mean(delays)), 1) if delays else "NA",
                    "median_delay": round(float(np.median(delays)), 1)
                    if delays
                    else "NA",
                }
            )
    return rows


def build_tcie_calibration_rows(
    results: list[TPTActiveBenchmarkResult],
    lambdas: list[float],
    thresholds: list[float],
) -> SummaryRows:
    rows: list[dict[str, str | float | int]] = []
    grouped: dict[float, list[TPTActiveBenchmarkResult]] = {
        lambda_value: [] for lambda_value in lambdas
    }
    for result in results:
        grouped.setdefault(result.config.effort_penalty_lambda, []).append(result)

    for lambda_value in lambdas:
        lambda_results = grouped.get(lambda_value, [])
        if not lambda_results:
            continue
        for threshold in thresholds:
            masking_delays: list[float] = []
            collapse_delays: list[float] = []
            healthy_false_positives = 0
            masking_detections = 0
            collapse_detections = 0

            for result in lambda_results:
                masking_warnings = threshold_crossings(result.tcie, threshold)
                healthy_false_positives += sum(
                    1
                    for warning in masking_warnings
                    if warning < result.config.masking_start
                )
                masking_summary = summarize_onset(
                    masking_warnings, result.config.masking_start
                )
                collapse_summary = summarize_onset(
                    masking_warnings, result.config.collapse_start
                )
                masking_delay = masking_summary.delay
                collapse_delay = collapse_summary.delay
                if masking_delay is not None:
                    masking_detections += 1
                    masking_delays.append(float(masking_delay))
                if collapse_delay is not None:
                    collapse_detections += 1
                    collapse_delays.append(float(collapse_delay))

            rows.append(
                {
                    "lambda": round(lambda_value, 3),
                    "threshold": round(threshold, 3),
                    "n_runs": len(lambda_results),
                    "masking_rate": round(masking_detections / len(lambda_results), 3),
                    "masking_median_delay": round(float(np.median(masking_delays)), 1)
                    if masking_delays
                    else "NA",
                    "collapse_rate": round(
                        collapse_detections / len(lambda_results), 3
                    ),
                    "collapse_median_delay": round(float(np.median(collapse_delays)), 1)
                    if collapse_delays
                    else "NA",
                    "healthy_false_positives": healthy_false_positives,
                    "mean_healthy_false_positives": round(
                        healthy_false_positives / len(lambda_results), 3
                    ),
                }
            )
    return rows


def build_masking_summary_rows(
    results: dict[str, TPTResult], tail_window: int = 60
) -> SummaryRows:
    rows: list[dict[str, str | float | int]] = []
    for regime in ("passive", "coercive"):
        result = results[regime]
        summary = summarize_result(result, tail_window=tail_window)
        rows.append(
            {
                "regime": regime,
                "condition": result.condition,
                "influence": round(result.config.influence, 3),
                "tail_abs_error": round(summary["mean_abs_error"], 3),
                "tail_action_gap": round(summary["mean_action_gap"], 3),
                "tail_effort": round(summary["mean_effort"], 3),
                "tail_sigma_p": round(summary["mean_sigma_p"], 3),
                "tail_sigma_p_eff": round(summary["mean_sigma_p_eff"], 3),
                "tail_sigma_a": round(summary["mean_sigma_a"], 3),
                "tail_sigma_phi": round(summary["mean_sigma_phi"], 3),
                "tail_tci": round(summary["mean_tci"], 3),
                "tail_tcie": round(summary["mean_tcie"], 3),
            }
        )
    return rows


def build_masking_grid_rows(
    config: TPTConfig,
    influences: list[float],
    lambdas: list[float],
    seeds: list[int],
    tail_window: int = 60,
) -> tuple[SummaryRows, SummaryRows]:
    raw_rows: list[dict[str, str | float | int]] = []
    numeric_fields = [
        "tail_abs_error",
        "tail_action_gap",
        "tail_effort",
        "tail_sigma_p",
        "tail_sigma_p_eff",
        "tail_sigma_a",
        "tail_sigma_phi",
        "tail_tci",
        "tail_tcie",
    ]

    for influence in influences:
        for penalty in lambdas:
            for seed in seeds:
                run_config = TPTConfig(
                    steps=config.steps,
                    particles=config.particles,
                    seed=seed,
                    drift=config.drift,
                    influence=influence,
                    process_scale=config.process_scale,
                    observation_scale=config.observation_scale,
                    observation_df=config.observation_df,
                    resample_threshold=config.resample_threshold,
                    prior_mean=config.prior_mean,
                    prior_scale=config.prior_scale,
                    condition="fm1",
                    actuation_noise_scale=config.actuation_noise_scale,
                    fm1_sigma_phi_level=config.fm1_sigma_phi_level,
                    fm3_sigma_phi_floor=config.fm3_sigma_phi_floor,
                    effort_penalty_lambda=penalty,
                    effort_floor=config.effort_floor,
                )
                results = run_coercive_masking_experiment(
                    run_config,
                    active_influence=influence,
                )
                for regime, result in results.items():
                    summary = summarize_result(result, tail_window=tail_window)
                    raw_rows.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "condition": result.condition,
                            "influence": round(influence, 3),
                            "lambda": round(penalty, 3),
                            "tail_abs_error": round(summary["mean_abs_error"], 6),
                            "tail_action_gap": round(summary["mean_action_gap"], 6),
                            "tail_effort": round(summary["mean_effort"], 6),
                            "tail_sigma_p": round(summary["mean_sigma_p"], 6),
                            "tail_sigma_p_eff": round(summary["mean_sigma_p_eff"], 6),
                            "tail_sigma_a": round(summary["mean_sigma_a"], 6),
                            "tail_sigma_phi": round(summary["mean_sigma_phi"], 6),
                            "tail_tci": round(summary["mean_tci"], 6),
                            "tail_tcie": round(summary["mean_tcie"], 6),
                        }
                    )

    grouped: dict[tuple[str, float, float], dict[str, object]] = {}
    for row in raw_rows:
        key = (str(row["regime"]), float(row["influence"]), float(row["lambda"]))
        if key not in grouped:
            grouped[key] = {
                "regime": row["regime"],
                "influence": row["influence"],
                "lambda": row["lambda"],
                "n_seeds": 0,
                **{field: [] for field in numeric_fields},
            }
        grouped_row = grouped[key]
        grouped_row["n_seeds"] = int(cast(int, grouped_row["n_seeds"])) + 1
        for field in numeric_fields:
            values = cast(list[float], grouped_row[field])
            assert isinstance(values, list)
            values.append(float(row[field]))

    summary_rows: list[dict[str, str | float | int]] = []
    for key in sorted(grouped):
        grouped_row = grouped[key]
        output_row: dict[str, object] = {
            "regime": str(grouped_row["regime"]),
            "influence": grouped_row["influence"],
            "lambda": grouped_row["lambda"],
            "n_seeds": grouped_row["n_seeds"],
        }
        for field in numeric_fields:
            values = np.asarray(cast(list[float], grouped_row[field]), dtype=float)
            output_row[f"mean_{field}"] = round(float(np.mean(values)), 3)
            output_row[f"std_{field}"] = round(float(np.std(values)), 3)
        if output_row["regime"] == "coercive":
            tail_tci = cast(float, output_row["mean_tail_tci"])
            tail_tcie = cast(float, output_row["mean_tail_tcie"])
            output_row["mean_masking_gap"] = round(
                tail_tci - tail_tcie,
                3,
            )
        else:
            output_row["mean_masking_gap"] = 0.0
        summary_rows.append(cast(dict[str, str | float | int], output_row))

    return raw_rows, summary_rows
