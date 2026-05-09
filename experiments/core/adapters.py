from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drift.umr import calibrate_umr_constant

from .detectors import run_river_drift_detector, run_umr_drift_detector
from .horizon_baselines import (
    HorizonBaselineResult,
    UMRRegulatorResult,
    compute_umr_regulator,
    run_ewma_baseline,
    run_fixed_window_baseline,
    run_melo_style_baseline,
    run_window_dilemma_baseline,
)


@dataclass(slots=True)
class ArenaAdapterConfig:
    fixed_window: int
    fixed_short_window: int
    fixed_long_window: int
    ewma_alpha: float
    block_size: int
    ema_alpha: float
    min_window: int
    max_window: int
    scale: float
    baseline_window: int
    prefix_length: int
    adwin_delta: float
    page_hinkley_delta: float
    page_hinkley_threshold: float
    page_hinkley_alpha: float
    window_dilemma_windows: tuple[int, int, int]
    window_dilemma_low_quantile: float
    window_dilemma_high_quantile: float
    melo_expert_windows: tuple[int, int, int, int, int]
    melo_learning_rate: float
    melo_discount: float


@dataclass(slots=True)
class ArenaAdapterResult:
    strategies: dict[str, HorizonBaselineResult]
    regulator: UMRRegulatorResult
    residual_signal: np.ndarray
    adwin_warnings: list[int]
    adwin_umr_warnings: list[int]
    adwin_umr_widths: np.ndarray
    adwin_umr_n_star: np.ndarray
    adwin_umr_drift_estimate: np.ndarray
    adwin_umr_cap_events: list[int]


def run_horizon_adapter(
    values: np.ndarray,
    *,
    config: ArenaAdapterConfig,
) -> tuple[dict[str, HorizonBaselineResult], UMRRegulatorResult]:
    regulator = compute_umr_regulator(
        values,
        block_size=config.block_size,
        ema_alpha=config.ema_alpha,
        baseline_window=config.baseline_window,
        prefix_length=config.prefix_length,
        scale=config.scale,
        min_window=config.min_window,
        max_window=config.max_window,
    )
    horizon_cap = regulator.window_sizes
    strategies = {
        "fixed_50": run_fixed_window_baseline(values, window=config.fixed_short_window),
        "fixed_100": run_fixed_window_baseline(values, window=config.fixed_window),
        "fixed_300": run_fixed_window_baseline(values, window=config.fixed_long_window),
        "ewma": run_ewma_baseline(values, alpha=config.ewma_alpha),
        "ewma_umr": run_ewma_baseline(
            values, alpha=config.ewma_alpha, horizon_cap=horizon_cap
        ),
        "umr": run_fixed_window_baseline(
            values, window=config.max_window, horizon_cap=horizon_cap
        ),
        "window_dilemma": run_window_dilemma_baseline(
            values,
            candidate_windows=config.window_dilemma_windows,
            block_size=config.block_size,
            ema_alpha=config.ema_alpha,
            baseline_window=config.baseline_window,
            prefix_length=config.prefix_length,
            low_quantile=config.window_dilemma_low_quantile,
            high_quantile=config.window_dilemma_high_quantile,
        ),
        "window_dilemma_umr": run_window_dilemma_baseline(
            values,
            candidate_windows=config.window_dilemma_windows,
            block_size=config.block_size,
            ema_alpha=config.ema_alpha,
            baseline_window=config.baseline_window,
            prefix_length=config.prefix_length,
            low_quantile=config.window_dilemma_low_quantile,
            high_quantile=config.window_dilemma_high_quantile,
            horizon_cap=horizon_cap,
        ),
        "melo": run_melo_style_baseline(
            values,
            expert_windows=config.melo_expert_windows,
            learning_rate=config.melo_learning_rate,
            discount=config.melo_discount,
        ),
        "melo_umr": run_melo_style_baseline(
            values,
            expert_windows=config.melo_expert_windows,
            learning_rate=config.melo_learning_rate,
            discount=config.melo_discount,
            horizon_cap=horizon_cap,
        ),
    }
    return strategies, regulator


def run_detector_adapter(
    signal: np.ndarray,
    *,
    config: ArenaAdapterConfig,
) -> list[int]:
    return run_river_drift_detector(
        signal,
        "ADWIN",
        adwin_delta=config.adwin_delta,
        page_hinkley_delta=config.page_hinkley_delta,
        page_hinkley_threshold=config.page_hinkley_threshold,
        page_hinkley_alpha=config.page_hinkley_alpha,
        kswin_window_size=30,
        kswin_stat_size=10,
        kswin_alpha=0.001,
    )


def run_regulated_detector_adapter(
    signal: np.ndarray,
    *,
    config: ArenaAdapterConfig,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    detector_ck = calibrate_umr_constant(list(signal[: config.prefix_length]))
    return run_umr_drift_detector(
        signal,
        delta=config.adwin_delta,
        Ck=detector_ck,
        drift_window=config.block_size,
        ema_alpha=config.ema_alpha,
        n_min=config.min_window,
        n_max=config.max_window,
    )
