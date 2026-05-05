from __future__ import annotations

import numpy as np
from river import drift


def run_river_drift_detector(
    signal: np.ndarray,
    detector_name: str,
    *,
    adwin_delta: float,
    page_hinkley_delta: float,
    page_hinkley_threshold: float,
    page_hinkley_alpha: float,
    kswin_window_size: int,
    kswin_stat_size: int,
    kswin_alpha: float,
) -> list[int]:
    if detector_name == "ADWIN":
        detector = drift.ADWIN(delta=adwin_delta)
    elif detector_name == "PageHinkley":
        detector = drift.PageHinkley(
            delta=page_hinkley_delta,
            threshold=page_hinkley_threshold,
            alpha=page_hinkley_alpha,
            mode="both",
        )
    elif detector_name == "KSWIN":
        detector = drift.KSWIN(
            window_size=kswin_window_size,
            stat_size=kswin_stat_size,
            alpha=kswin_alpha,
        )
    elif detector_name == "NoDrift":
        detector = drift.NoDrift()
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    warnings: list[int] = []
    for index, value in enumerate(signal):
        detector.update(float(value))
        if detector.drift_detected:
            warnings.append(index)
    return warnings


def create_river_drift_detector(
    detector_name: str,
    *,
    adwin_delta: float,
    page_hinkley_delta: float,
    page_hinkley_threshold: float,
    page_hinkley_alpha: float,
    kswin_window_size: int,
    kswin_stat_size: int,
    kswin_alpha: float,
):
    if detector_name == "ADWIN":
        return drift.ADWIN(delta=adwin_delta)
    if detector_name == "PageHinkley":
        return drift.PageHinkley(
            delta=page_hinkley_delta,
            threshold=page_hinkley_threshold,
            alpha=page_hinkley_alpha,
            mode="both",
        )
    if detector_name == "KSWIN":
        return drift.KSWIN(
            window_size=kswin_window_size,
            stat_size=kswin_stat_size,
            alpha=kswin_alpha,
        )
    if detector_name == "NoDrift":
        return drift.NoDrift()
    raise ValueError(f"Unknown detector: {detector_name}")
