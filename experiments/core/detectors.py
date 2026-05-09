from __future__ import annotations

from collections import deque

import numpy as np
from river import drift

from drift.umr import UsefulMemoryRegulator


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


def run_umr_drift_detector(
    signal: np.ndarray,
    *,
    delta: float,
    Ck: float,
    drift_window: int,
    ema_alpha: float,
    n_min: int,
    n_max: int,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    regulator = UsefulMemoryRegulator(
        Ck=Ck,
        drift_window=drift_window,
        ema_alpha=ema_alpha,
        n_min=n_min,
        n_max=n_max,
    )
    detector = drift.ADWIN(delta=delta)
    buffer: deque[float] = deque(maxlen=n_max)
    warnings: list[int] = []
    cap_events: list[int] = []
    widths = np.zeros(signal.size, dtype=float)
    n_star = np.zeros(signal.size, dtype=float)
    drift_estimate = np.zeros(signal.size, dtype=float)
    estimates = np.zeros(signal.size, dtype=float)

    for index, value in enumerate(signal):
        x = float(value)
        regulator.observe(x)
        buffer.append(x)
        detector.update(x)
        current_width = float(detector.width)
        current_limit = float(regulator.n_star)
        drift_estimate[index] = float(regulator.delta_est)
        n_star[index] = current_limit

        if current_width > current_limit:
            cap_events.append(index)
            detector = drift.ADWIN(delta=delta)
            recent = list(buffer)[-int(current_limit) :]
            for recent_value in recent:
                detector.update(recent_value)
            current_width = float(detector.width)

        widths[index] = current_width
        estimates[index] = float(detector.estimation)
        if detector.drift_detected:
            warnings.append(index)

    return warnings, widths, n_star, drift_estimate, cap_events, estimates
