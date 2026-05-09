from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import (
    ArenaAdapterConfig,
    run_detector_adapter,
    run_horizon_adapter,
    run_regulated_detector_adapter,
)
from .horizon_baselines import HorizonBaselineResult, UMRRegulatorResult


@dataclass(slots=True)
class HorizonArenaResult:
    strategies: dict[str, HorizonBaselineResult]
    regulator: UMRRegulatorResult
    residual_signal: np.ndarray
    adwin_warnings: list[int]
    adwin_umr_warnings: list[int]
    adwin_umr_widths: np.ndarray
    adwin_umr_n_star: np.ndarray
    adwin_umr_drift_estimate: np.ndarray
    adwin_umr_cap_events: list[int]


def build_horizon_arena(
    values: np.ndarray,
    *,
    fixed_window: int,
    fixed_short_window: int,
    fixed_long_window: int,
    ewma_alpha: float,
    block_size: int,
    ema_alpha: float,
    min_window: int,
    max_window: int,
    scale: float,
    baseline_window: int,
    prefix_length: int,
    adwin_delta: float,
    page_hinkley_delta: float,
    page_hinkley_threshold: float,
    page_hinkley_alpha: float,
    window_dilemma_windows: tuple[int, int, int],
    window_dilemma_low_quantile: float,
    window_dilemma_high_quantile: float,
    melo_expert_windows: tuple[int, int, int, int, int],
    melo_learning_rate: float,
    melo_discount: float,
) -> HorizonArenaResult:
    adapter_config = ArenaAdapterConfig(
        fixed_window=fixed_window,
        fixed_short_window=fixed_short_window,
        fixed_long_window=fixed_long_window,
        ewma_alpha=ewma_alpha,
        block_size=block_size,
        ema_alpha=ema_alpha,
        min_window=min_window,
        max_window=max_window,
        scale=scale,
        baseline_window=baseline_window,
        prefix_length=prefix_length,
        adwin_delta=adwin_delta,
        page_hinkley_delta=page_hinkley_delta,
        page_hinkley_threshold=page_hinkley_threshold,
        page_hinkley_alpha=page_hinkley_alpha,
        window_dilemma_windows=window_dilemma_windows,
        window_dilemma_low_quantile=window_dilemma_low_quantile,
        window_dilemma_high_quantile=window_dilemma_high_quantile,
        melo_expert_windows=melo_expert_windows,
        melo_learning_rate=melo_learning_rate,
        melo_discount=melo_discount,
    )
    strategies, regulator = run_horizon_adapter(values, config=adapter_config)
    residual_signal = np.abs(strategies["fixed_100"].estimate - values)
    residual_signal = np.nan_to_num(residual_signal, nan=0.0)
    adwin_warnings = run_detector_adapter(residual_signal, config=adapter_config)
    (
        adwin_umr_warnings,
        adwin_umr_widths,
        adwin_umr_n_star,
        adwin_umr_drift_estimate,
        adwin_umr_cap_events,
        _,
    ) = run_regulated_detector_adapter(residual_signal, config=adapter_config)
    return HorizonArenaResult(
        strategies=strategies,
        regulator=regulator,
        residual_signal=residual_signal,
        adwin_warnings=adwin_warnings,
        adwin_umr_warnings=adwin_umr_warnings,
        adwin_umr_widths=adwin_umr_widths,
        adwin_umr_n_star=adwin_umr_n_star,
        adwin_umr_drift_estimate=adwin_umr_drift_estimate,
        adwin_umr_cap_events=adwin_umr_cap_events,
    )
