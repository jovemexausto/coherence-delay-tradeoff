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
from ..core.umr_arena import build_horizon_arena


@dataclass(slots=True)
class Elec2Config:
    demand_key: str = "nswdemand"
    page_hinkley_delta: float = 0.01
    page_hinkley_threshold: float = 84.0
    page_hinkley_alpha: float = 0.9999
    warning_threshold: float = 0.295
    max_gap: int = 5000
    fixed_window: int = 100
    ewma_alpha: float = 0.05
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
    window_dilemma_windows: tuple[int, int, int] = (50, 100, 300)
    window_dilemma_low_quantile: float = 0.33
    window_dilemma_high_quantile: float = 0.67
    melo_expert_windows: tuple[int, int, int, int, int] = (30, 50, 100, 200, 300)
    melo_learning_rate: float = 6.0
    melo_discount: float = 0.995


@dataclass(slots=True)
class Elec2DetectionResult:
    sigma: np.ndarray
    estimate: np.ndarray
    window_sizes: np.ndarray
    warnings: list[int]
    matched_warnings: list[int]
    matched_events: list[int]
    lead_times: list[int]


@dataclass(slots=True)
class Elec2ExperimentResult:
    config: Elec2Config
    values: np.ndarray
    events: list[int]
    fixed_100: Elec2DetectionResult
    fixed_50: Elec2DetectionResult
    fixed_300: Elec2DetectionResult
    ewma: Elec2DetectionResult
    dynamic: Elec2DetectionResult
    window_dilemma: Elec2DetectionResult
    melo: Elec2DetectionResult
    adwin: Elec2DetectionResult
    adwin_umr: Elec2DetectionResult
    cusum: ScalarDetectionResult
    rls: ScalarDetectionResult
    kalman: ScalarDetectionResult
    frechet: ScalarDetectionResult
    mmd: ScalarDetectionResult
    arena: dict[str, Elec2DetectionResult]
    residual_signal: np.ndarray
    dynamic_drift_estimate: np.ndarray


def load_elec2_values(config: Elec2Config | None = None) -> np.ndarray:
    config = config or Elec2Config()
    stream = cast(Iterable[tuple[dict[str, float], object]], datasets.Elec2())
    values = np.fromiter(
        (float(features[config.demand_key]) for features, _ in stream),
        dtype=float,
    )
    return (values - values.mean()) / values.std()


def detect_page_hinkley_events(
    values: np.ndarray,
    config: Elec2Config,
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
    config: Elec2Config,
) -> Elec2DetectionResult:
    warnings = _extract_warnings(sigma, config.warning_threshold)
    match_result = match_warnings_to_events(warnings, events, config.max_gap)
    return Elec2DetectionResult(
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
    config: Elec2Config,
    *,
    window_sizes: np.ndarray | None = None,
) -> Elec2DetectionResult:
    match_result = match_warnings_to_events(warnings, events, config.max_gap)
    return Elec2DetectionResult(
        sigma=signal,
        estimate=np.full(signal.size, np.nan),
        window_sizes=np.full(signal.size, np.nan)
        if window_sizes is None
        else window_sizes,
        warnings=warnings,
        matched_warnings=match_result.matched_warnings,
        matched_events=match_result.matched_events,
        lead_times=match_result.lead_times,
    )


def run_elec2_experiments(config: Elec2Config | None = None) -> Elec2ExperimentResult:
    config = config or Elec2Config()
    values = load_elec2_values(config)
    events = detect_page_hinkley_events(values, config)
    adaptive_gap = (
        max(1, int(np.median(np.diff(events)) * 0.5))
        if len(events) > 1
        else config.max_gap
    )
    config.max_gap = min(config.max_gap, adaptive_gap)

    arena_result = build_horizon_arena(
        values,
        fixed_window=config.fixed_window,
        fixed_short_window=50,
        fixed_long_window=300,
        ewma_alpha=config.ewma_alpha,
        block_size=config.dynamic_window_delta,
        ema_alpha=config.dynamic_alpha,
        min_window=config.dynamic_min_window,
        max_window=config.dynamic_max_window,
        scale=config.dynamic_scale,
        baseline_window=config.dynamic_baseline_window,
        prefix_length=config.baseline_prefix_length,
        adwin_delta=config.adwin_delta,
        page_hinkley_delta=config.page_hinkley_delta,
        page_hinkley_threshold=config.page_hinkley_threshold,
        page_hinkley_alpha=config.page_hinkley_alpha,
        window_dilemma_windows=config.window_dilemma_windows,
        window_dilemma_low_quantile=config.window_dilemma_low_quantile,
        window_dilemma_high_quantile=config.window_dilemma_high_quantile,
        melo_expert_windows=config.melo_expert_windows,
        melo_learning_rate=config.melo_learning_rate,
        melo_discount=config.melo_discount,
    )
    residual_signal = arena_result.residual_signal

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

    return Elec2ExperimentResult(
        config=config,
        values=values,
        events=events,
        fixed_100=_build_detection_result(
            arena_result.strategies["fixed_100"].sigma,
            arena_result.strategies["fixed_100"].estimate,
            arena_result.strategies["fixed_100"].window_sizes,
            events,
            config,
        ),
        fixed_50=_build_detection_result(
            arena_result.strategies["fixed_50"].sigma,
            arena_result.strategies["fixed_50"].estimate,
            arena_result.strategies["fixed_50"].window_sizes,
            events,
            config,
        ),
        fixed_300=_build_detection_result(
            arena_result.strategies["fixed_300"].sigma,
            arena_result.strategies["fixed_300"].estimate,
            arena_result.strategies["fixed_300"].window_sizes,
            events,
            config,
        ),
        ewma=_build_detection_result(
            arena_result.strategies["ewma"].sigma,
            arena_result.strategies["ewma"].estimate,
            arena_result.strategies["ewma"].window_sizes,
            events,
            config,
        ),
        dynamic=_build_detection_result(
            arena_result.strategies["umr"].sigma,
            arena_result.strategies["umr"].estimate,
            arena_result.strategies["umr"].window_sizes,
            events,
            config,
        ),
        window_dilemma=_build_detection_result(
            arena_result.strategies["window_dilemma"].sigma,
            arena_result.strategies["window_dilemma"].estimate,
            arena_result.strategies["window_dilemma"].window_sizes,
            events,
            config,
        ),
        melo=_build_detection_result(
            arena_result.strategies["melo"].sigma,
            arena_result.strategies["melo"].estimate,
            arena_result.strategies["melo"].window_sizes,
            events,
            config,
        ),
        adwin=_build_detection_result_from_warnings(
            residual_signal,
            arena_result.adwin_warnings,
            events,
            config,
        ),
        adwin_umr=_build_detection_result_from_warnings(
            residual_signal,
            arena_result.adwin_umr_warnings,
            events,
            config,
            window_sizes=arena_result.adwin_umr_widths,
        ),
        cusum=cusum,
        rls=rls,
        kalman=kalman,
        frechet=frechet,
        mmd=mmd,
        arena={
            name: _build_detection_result(
                arena_result.strategies[name].sigma,
                arena_result.strategies[name].estimate,
                arena_result.strategies[name].window_sizes,
                events,
                config,
            )
            for name in (
                "ewma",
                "ewma_umr",
                "window_dilemma",
                "window_dilemma_umr",
                "melo",
                "melo_umr",
            )
        }
        | {
            "adwin": _build_detection_result_from_warnings(
                residual_signal,
                arena_result.adwin_warnings,
                events,
                config,
            ),
            "adwin_umr": _build_detection_result_from_warnings(
                residual_signal,
                arena_result.adwin_umr_warnings,
                events,
                config,
                window_sizes=arena_result.adwin_umr_widths,
            ),
        },
        residual_signal=residual_signal,
        dynamic_drift_estimate=arena_result.regulator.drift_estimate,
    )
