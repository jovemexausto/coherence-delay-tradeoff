from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import cast

import numpy as np
from river import datasets

from ..core.baselines import (
    ScalarDetectionResult,
    run_frechet_detector,
    run_cusum_detector,
    run_forgetting_factor_rls_detector,
    run_scalar_kalman_detector,
    run_mmd_detector,
)
from ..core.common import match_warnings_to_events
from ..core.detectors import run_river_drift_detector


@dataclass(slots=True)
class BikesConfig:
    page_hinkley_delta: float = 0.01
    page_hinkley_threshold: float = 84.0
    page_hinkley_alpha: float = 0.9999
    warning_threshold: float = 0.295
    max_gap: int = 5000
    fixed_window: int = 100
    dynamic_alpha: float = 0.03
    dynamic_window_delta: int = 24
    dynamic_min_window: int = 30
    dynamic_max_window: int = 300
    dynamic_scale: float = 1.25
    dynamic_baseline_window: int = 100
    adwin_delta: float = 0.03
    baseline_prefix_length: int = 2000
    baseline_warning_threshold: float = 0.295
    frechet_window_size: int = 100
    mmd_window_size: int = 100
    rls_forgetting_factor: float = 0.995
    kalman_process_scale: float = 0.02
    cusum_drift_allowance: float = 0.25
    cusum_alarm_scale: float = 8.0


@dataclass(slots=True)
class BikesDetectionResult:
    sigma: np.ndarray
    estimate: np.ndarray
    window_sizes: np.ndarray
    warnings: list[int]
    matched_warnings: list[int]
    matched_events: list[int]
    lead_times: list[int]


@dataclass(slots=True)
class BikesExperimentResult:
    config: BikesConfig
    values: np.ndarray
    events: list[int]
    fixed_100: BikesDetectionResult
    fixed_50: BikesDetectionResult
    fixed_300: BikesDetectionResult
    dynamic: BikesDetectionResult
    adwin: BikesDetectionResult
    cusum: ScalarDetectionResult
    rls: ScalarDetectionResult
    kalman: ScalarDetectionResult
    frechet: ScalarDetectionResult
    mmd: ScalarDetectionResult
    residual_signal: np.ndarray
    dynamic_drift_estimate: np.ndarray


def load_bikes_values(config: BikesConfig | None = None) -> np.ndarray:
    config = config or BikesConfig()
    stream = cast(Iterable[tuple[object, float]], datasets.Bikes())
    values = np.fromiter((float(target) for _, target in stream), dtype=float)
    return (values - values.mean()) / values.std()


def detect_page_hinkley_events(
    values: np.ndarray,
    config: BikesConfig,
) -> list[int]:
    return run_river_drift_detector(
        values,
        "PageHinkley",
        adwin_delta=config.adwin_delta,
        page_hinkley_delta=config.page_hinkley_delta,
        page_hinkley_threshold=config.page_hinkley_threshold,
        page_hinkley_alpha=config.page_hinkley_alpha,
        kswin_window_size=30,
        kswin_stat_size=10,
        kswin_alpha=0.001,
    )


def _compute_sigma_fixed(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    windows = np.full(values.size, float(window))
    for index in range(window - 1, values.size):
        mean_value = float(np.mean(values[index - window + 1 : index + 1]))
        estimate[index] = mean_value
        sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)
    return sigma, estimate, windows


def _compute_sigma_dynamic(
    values: np.ndarray,
    config: BikesConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    windows = np.full(values.size, float(config.dynamic_baseline_window))
    zeta_hat = np.zeros(values.size)

    d = config.dynamic_window_delta
    ck = float(
        np.mean(np.abs(values[:2000] - np.mean(values[:2000])))
        * np.sqrt(config.dynamic_baseline_window)
        * config.dynamic_scale
    )
    ema = 0.0

    for index in range(values.size):
        if index >= 2 * d:
            recent_mean = float(np.mean(values[index - d : index]))
            previous_mean = float(np.mean(values[index - 2 * d : index - d]))
            local_drift = abs(recent_mean - previous_mean) / d
            ema = (
                config.dynamic_alpha * local_drift + (1.0 - config.dynamic_alpha) * ema
            )
            zeta_hat[index] = ema
            windows[index] = np.clip(
                (ck / max(ema, 1e-9)) ** (2.0 / 3.0),
                config.dynamic_min_window,
                config.dynamic_max_window,
            )
        window = int(round(windows[index]))
        if index >= window - 1:
            mean_value = float(np.mean(values[index - window + 1 : index + 1]))
            estimate[index] = mean_value
            sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)

    return sigma, estimate, windows, zeta_hat


def _extract_warnings(sigma: np.ndarray, threshold: float) -> list[int]:
    warnings: list[int] = []
    below = False
    for index, value in enumerate(sigma):
        if np.isnan(value):
            continue
        if value < threshold and not below:
            warnings.append(index)
            below = True
        elif value >= threshold:
            below = False
    return warnings


def _build_detection_result(
    sigma: np.ndarray,
    estimate: np.ndarray,
    windows: np.ndarray,
    events: list[int],
    config: BikesConfig,
) -> BikesDetectionResult:
    warnings = _extract_warnings(sigma, config.warning_threshold)
    match_result = match_warnings_to_events(warnings, events, config.max_gap)
    return BikesDetectionResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=windows,
        warnings=warnings,
        matched_warnings=match_result.matched_warnings,
        matched_events=match_result.matched_events,
        lead_times=match_result.lead_times,
    )


def _build_detection_result_from_warnings(
    signal: np.ndarray,
    warnings: list[int],
    events: list[int],
    config: BikesConfig,
) -> BikesDetectionResult:
    match_result = match_warnings_to_events(warnings, events, config.max_gap)
    return BikesDetectionResult(
        sigma=signal,
        estimate=np.full(signal.size, np.nan),
        window_sizes=np.full(signal.size, np.nan),
        warnings=warnings,
        matched_warnings=match_result.matched_warnings,
        matched_events=match_result.matched_events,
        lead_times=match_result.lead_times,
    )


def run_bikes_experiments(config: BikesConfig | None = None) -> BikesExperimentResult:
    config = config or BikesConfig()
    values = load_bikes_values(config)
    events = detect_page_hinkley_events(values, config)
    adaptive_gap = (
        max(1, int(np.median(np.diff(events)) * 0.5))
        if len(events) > 1
        else config.max_gap
    )
    config.max_gap = min(config.max_gap, adaptive_gap)

    sigma_100, estimate_100, windows_100 = _compute_sigma_fixed(
        values, config.fixed_window
    )
    sigma_50, estimate_50, windows_50 = _compute_sigma_fixed(values, 50)
    sigma_300, estimate_300, windows_300 = _compute_sigma_fixed(values, 300)
    sigma_dynamic, estimate_dynamic, windows_dynamic, zeta_hat = _compute_sigma_dynamic(
        values,
        config,
    )
    residual_signal = np.abs(estimate_100 - values)
    residual_signal = np.nan_to_num(residual_signal, nan=0.0)
    adwin_warnings = run_river_drift_detector(
        residual_signal,
        "ADWIN",
        adwin_delta=config.adwin_delta,
        page_hinkley_delta=config.page_hinkley_delta,
        page_hinkley_threshold=config.page_hinkley_threshold,
        page_hinkley_alpha=config.page_hinkley_alpha,
        kswin_window_size=30,
        kswin_stat_size=10,
        kswin_alpha=0.001,
    )

    cusum = run_cusum_detector(
        values,
        events,
        config.max_gap,
        warning_threshold=config.baseline_warning_threshold,
        prefix_length=config.baseline_prefix_length,
        drift_allowance=config.cusum_drift_allowance,
        alarm_scale=config.cusum_alarm_scale,
    )
    rls = run_forgetting_factor_rls_detector(
        values,
        events,
        config.max_gap,
        warning_threshold=config.baseline_warning_threshold,
        prefix_length=config.baseline_prefix_length,
        forgetting_factor=config.rls_forgetting_factor,
    )
    kalman = run_scalar_kalman_detector(
        values,
        events,
        config.max_gap,
        warning_threshold=config.baseline_warning_threshold,
        prefix_length=config.baseline_prefix_length,
        process_scale=config.kalman_process_scale,
    )
    frechet = run_frechet_detector(
        values,
        events,
        config.max_gap,
        warning_threshold=config.baseline_warning_threshold,
        prefix_length=config.baseline_prefix_length,
        window_size=config.frechet_window_size,
    )
    mmd = run_mmd_detector(
        values,
        events,
        config.max_gap,
        warning_threshold=config.baseline_warning_threshold,
        prefix_length=config.baseline_prefix_length,
        window_size=config.mmd_window_size,
    )

    return BikesExperimentResult(
        config=config,
        values=values,
        events=events,
        fixed_100=_build_detection_result(
            sigma_100, estimate_100, windows_100, events, config
        ),
        fixed_50=_build_detection_result(
            sigma_50, estimate_50, windows_50, events, config
        ),
        fixed_300=_build_detection_result(
            sigma_300, estimate_300, windows_300, events, config
        ),
        dynamic=_build_detection_result(
            sigma_dynamic, estimate_dynamic, windows_dynamic, events, config
        ),
        adwin=_build_detection_result_from_warnings(
            residual_signal,
            adwin_warnings,
            events,
            config,
        ),
        cusum=cusum,
        rls=rls,
        kalman=kalman,
        frechet=frechet,
        mmd=mmd,
        residual_signal=residual_signal,
        dynamic_drift_estimate=zeta_hat,
    )
