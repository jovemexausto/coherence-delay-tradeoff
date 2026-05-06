from __future__ import annotations

import numpy as np

from .model import (
    ACTIVE_BASELINE_DETECTORS,
    KuaiRandUserDetectionResult,
)
from ..core.types import SummaryRow, SummaryRows


def build_kuairand_summary_rows(
    results: list[KuaiRandUserDetectionResult],
) -> SummaryRows:
    rows: list[SummaryRow] = []
    phase_labels = {
        "masking_detection": "bubble_detection",
        "collapse_detection": "collapse_detection",
    }
    for phase in ("masking_detection", "collapse_detection"):
        for detector in ("TCI", "TCIE", "TCIE-EWMA", *ACTIVE_BASELINE_DETECTORS):
            detections = 0
            delays: list[float] = []
            for result in results:
                summary = getattr(result, phase)[detector]
                if summary["detections"]:
                    detections += 1
                    if summary["median_delay"] is not None:
                        delays.append(float(summary["median_delay"]))
            rows.append(
                {
                    "phase": phase_labels[phase],
                    "detector": detector,
                    "n_users": len(results),
                    "detections": detections,
                    "rate": round(detections / len(results), 3) if results else 0.0,
                    "median_delay": round(float(np.median(delays)), 1)
                    if delays
                    else "NA",
                }
            )
    return rows
