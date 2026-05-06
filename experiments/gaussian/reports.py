from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import (
    SampleComplexityResult,
    SinkhornRuntimeResult,
    TGTResult,
    TGT_CONDITIONS,
    UCurveResult,
)
from ..core.types import SummaryRows


def summarize_tgt_result(result: TGTResult, tail_window: int = 100) -> dict[str, float]:
    tail = slice(-min(tail_window, result.config.steps), None)
    return {
        "mean_v_p": float(np.mean(result.v_p[tail])),
        "mean_v_a": float(np.mean(result.v_a[tail])),
        "mean_v_phi": float(np.mean(result.v_phi[tail])),
        "mean_sigma_p": float(np.mean(result.sigma_p[tail])),
        "mean_sigma_a": float(np.mean(result.sigma_a[tail])),
        "mean_sigma_phi": float(np.mean(result.sigma_phi[tail])),
        "mean_tci": float(np.mean(result.tci[tail])),
        "mean_v_total": float(np.mean(result.v_total[tail])),
    }


def build_ablation_rows(results: dict[str, TGTResult]) -> SummaryRows:
    rows: list[dict[str, str | float]] = []
    for condition in TGT_CONDITIONS:
        summary = summarize_tgt_result(results[condition])
        rows.append(
            {
                "condition": condition,
                "mean_v_p": round(summary["mean_v_p"], 3),
                "mean_v_a": round(summary["mean_v_a"], 3),
                "mean_v_phi": round(summary["mean_v_phi"], 3),
                "mean_sigma_p": round(summary["mean_sigma_p"], 3),
                "mean_sigma_a": round(summary["mean_sigma_a"], 3),
                "mean_sigma_phi": round(summary["mean_sigma_phi"], 3),
                "mean_tci": round(summary["mean_tci"], 3),
                "mean_v_total": round(summary["mean_v_total"], 3),
            }
        )
    return rows


def export_rows_csv(rows: SummaryRows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_ucurve_rows(result: UCurveResult) -> SummaryRows:
    rows: list[dict[str, float | int]] = []
    for drift, n_star, e_min in zip(
        result.drift_values,
        result.empirical_n_star,
        result.empirical_e_min,
        strict=True,
    ):
        rows.append(
            {
                "drift": round(float(drift), 4),
                "n_star": int(n_star),
                "e_min": round(float(e_min), 4),
                "scaled_constant": round(float(e_min / np.cbrt(drift)), 4),
            }
        )
    return rows


def build_sample_complexity_rows(
    result: SampleComplexityResult,
) -> SummaryRows:
    rows: list[dict[str, float | int]] = []
    for window_size, mean_error, std_error in zip(
        result.window_sizes,
        result.mean_absolute_error,
        result.std_absolute_error,
        strict=True,
    ):
        rows.append(
            {
                "window_size": int(window_size),
                "mean_absolute_error": round(float(mean_error), 4),
                "std_absolute_error": round(float(std_error), 4),
            }
        )
    return rows


def build_sinkhorn_runtime_rows(
    result: SinkhornRuntimeResult,
) -> SummaryRows:
    rows: list[dict[str, float | int]] = []
    for d_index, dimension in enumerate(result.dimensions):
        for n_index, window_size in enumerate(result.window_sizes):
            for e_index, epsilon in enumerate(result.epsilons):
                rows.append(
                    {
                        "dimension": int(dimension),
                        "window_size": int(window_size),
                        "epsilon": round(float(epsilon), 3),
                        "mean_runtime_ms": round(
                            float(result.mean_runtime_ms[d_index, n_index, e_index]),
                            3,
                        ),
                        "mean_abs_bias": round(
                            float(result.mean_abs_bias[d_index, n_index, e_index]),
                            4,
                        ),
                        "mean_iterations": round(
                            float(result.mean_iterations[d_index, n_index, e_index]),
                            1,
                        ),
                        "mean_pairwise_evals_per_s": round(
                            float(
                                result.mean_pairwise_evals_per_s[
                                    d_index, n_index, e_index
                                ]
                            ),
                            1,
                        ),
                    }
                )
    return rows
