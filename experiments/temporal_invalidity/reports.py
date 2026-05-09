from __future__ import annotations

import numpy as np

from .model import TemporalInvalidityResult


def build_temporal_invalidity_rows(
    result: TemporalInvalidityResult,
) -> list[dict[str, float | int | str]]:
    summaries = result.summaries
    cap_only_steps = float(
        np.mean([np.sum(trace.cap_only_mask) for trace in result.traces])
    )
    cap_only_segments = float(
        np.mean([len(_segments(trace.cap_only_mask)) for trace in result.traces])
    )
    cap_only_mean_umr = (
        float(
            np.mean(
                [
                    float(np.mean(trace.umr_width_policy[trace.cap_only_mask]))
                    for trace in result.traces
                    if np.any(trace.cap_only_mask)
                ]
            )
        )
        if any(np.any(trace.cap_only_mask) for trace in result.traces)
        else float("nan")
    )

    rows: list[dict[str, float | int | str]] = []
    for method, label in (
        ("fixed_400", "Fixed-400"),
        ("fixed_100", "Fixed-100"),
        ("adwin", "ADWIN"),
        ("umr", "UMR"),
    ):
        summary = summaries[method]
        rows.append(
            {
                "method": label,
                "global_accuracy": round(summary.global_accuracy_mean, 4),
                "global_log_loss": round(summary.global_log_loss_mean, 4),
                "cap_only_accuracy": round(summary.cap_only_accuracy_mean, 4),
                "cap_only_log_loss": round(summary.cap_only_log_loss_mean, 4),
                "delta_cap_only_vs_fixed_400_pp": round(
                    100.0
                    * (
                        summary.cap_only_accuracy_mean
                        - summaries["fixed_400"].cap_only_accuracy_mean
                    ),
                    2,
                ),
                "cap_only_steps": round(cap_only_steps, 1),
                "cap_only_segments": round(cap_only_segments, 1),
                "mean_umr_width_cap_only": round(cap_only_mean_umr, 2),
            }
        )
    return rows


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments
