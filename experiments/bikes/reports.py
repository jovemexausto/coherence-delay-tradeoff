from __future__ import annotations

import numpy as np

from .model import BikesExperimentResult
from ..core.types import DetectionSummaryLike


ARENA_BASELINES: tuple[str, ...] = (
    "ewma",
    "window_dilemma",
    "melo",
    "adwin",
)


def summarize_detection(
    result: DetectionSummaryLike,
) -> dict[str, float]:
    lead_times = np.asarray(result.lead_times, dtype=float)
    return {
        "warnings": float(len(result.warnings)),
        "leads": float(len(result.lead_times)),
        "precision": float(len(result.lead_times) / len(result.warnings))
        if result.warnings
        else 0.0,
        "median_lead": float(np.median(lead_times)) if lead_times.size else 0.0,
        "mean_lead": float(np.mean(lead_times)) if lead_times.size else 0.0,
        "min_lead": float(np.min(lead_times)) if lead_times.size else 0.0,
        "max_lead": float(np.max(lead_times)) if lead_times.size else 0.0,
    }


def build_bikes_rows(result: BikesExperimentResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, detection in (
        ("fixed_50", result.fixed_50),
        ("fixed_100", result.fixed_100),
        ("fixed_300", result.fixed_300),
        ("ewma", result.ewma),
        ("umr", result.dynamic),
        ("window_dilemma", result.window_dilemma),
        ("melo", result.melo),
        ("adwin", result.adwin),
        ("adwin_umr", result.adwin_umr),
        ("cusum", result.cusum),
        ("rls", result.rls),
        ("kalman", result.kalman),
        ("frechet", result.frechet),
        ("mmd", result.mmd),
    ):
        summary = summarize_detection(detection)
        rows.append(
            {
                "strategy": name,
                "warnings": int(summary["warnings"]),
                "leads": int(summary["leads"]),
                "precision": round(summary["precision"], 3),
                "median_lead": round(summary["median_lead"], 1),
                "mean_lead": round(summary["mean_lead"], 1),
                "min_lead": round(summary["min_lead"], 1),
                "max_lead": round(summary["max_lead"], 1),
            }
        )
    return rows


def build_bikes_arena_rows(
    result: BikesExperimentResult,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for baseline in ARENA_BASELINES:
        base_result = result.arena[baseline]
        regulated_result = result.arena[f"{baseline}_umr"]
        base_summary = summarize_detection(base_result)
        regulated_summary = summarize_detection(regulated_result)
        for condition, summary in (("base", base_summary), ("umr", regulated_summary)):
            rows.append(
                {
                    "baseline": baseline,
                    "condition": condition,
                    "strategy": baseline if condition == "base" else f"{baseline}_umr",
                    "warnings": int(summary["warnings"]),
                    "leads": int(summary["leads"]),
                    "precision": round(summary["precision"], 3),
                    "median_lead": round(summary["median_lead"], 1),
                    "mean_lead": round(summary["mean_lead"], 1),
                    "delta_precision_vs_base": round(
                        summary["precision"] - base_summary["precision"], 3
                    ),
                    "delta_median_lead_vs_base": round(
                        summary["median_lead"] - base_summary["median_lead"], 1
                    ),
                }
            )
    return rows
