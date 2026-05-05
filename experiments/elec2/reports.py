from __future__ import annotations

import numpy as np

from .model import Elec2DetectionResult, Elec2ExperimentResult


def summarize_detection(
    result: Elec2DetectionResult | object,
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


def build_elec2_rows(result: Elec2ExperimentResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, detection in (
        ("fixed_50", result.fixed_50),
        ("fixed_100", result.fixed_100),
        ("fixed_300", result.fixed_300),
        ("dynamic", result.dynamic),
        ("adwin", result.adwin),
        ("cusum", result.cusum),
        ("rls", result.rls),
        ("kalman", result.kalman),
        ("frechet", result.frechet),
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
