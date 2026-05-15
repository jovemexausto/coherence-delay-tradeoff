from __future__ import annotations

from .model import InvalidityGapResult


def build_gap_rows(result: InvalidityGapResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for summary in result.summaries:
        rows.append(
            {
                "detector_delta": round(float(summary.detector_delta), 4),
                "mean_t_valid": round(float(summary.mean_t_valid), 1),
                "mean_t_detect": round(float(summary.mean_t_detect), 1),
                "mean_gap": round(float(summary.mean_gap), 1),
                "std_gap": round(float(summary.std_gap), 1),
                "positive_gap_rate": round(float(summary.positive_gap_rate), 3),
                "detection_rate": round(float(summary.detection_rate), 3),
            }
        )
    return rows


def build_trace_rows(result: InvalidityGapResult) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for trace in result.traces:
        rows.append(
            {
                "seed": trace.seed,
                "t_valid": "" if trace.t_valid is None else trace.t_valid,
                "t_detect": "" if trace.t_detect is None else trace.t_detect,
                "invalidity_gap": ""
                if trace.invalidity_gap is None
                else trace.invalidity_gap,
            }
        )
    return rows
