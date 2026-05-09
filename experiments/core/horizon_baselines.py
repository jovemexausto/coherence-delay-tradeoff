from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class HorizonBaselineResult:
    sigma: np.ndarray
    estimate: np.ndarray
    window_sizes: np.ndarray
    drift_estimate: np.ndarray


@dataclass(slots=True)
class UMRRegulatorResult:
    window_sizes: np.ndarray
    drift_estimate: np.ndarray


def _prefix_sums(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros(values.size + 1, dtype=float)
    prefix[1:] = np.cumsum(values, dtype=float)
    return prefix


def _window_mean(prefix: np.ndarray, end_index: int, window: int) -> float:
    right = end_index + 1
    left = right - window
    return float((prefix[right] - prefix[left]) / window)


def _local_drift_signal(
    values: np.ndarray,
    *,
    block_size: int,
    ema_alpha: float,
) -> np.ndarray:
    prefix = _prefix_sums(values)
    drift = np.zeros(values.size, dtype=float)
    ema = 0.0
    for index in range(values.size):
        if index < 2 * block_size:
            continue
        recent_mean = _window_mean(prefix, index - 1, block_size)
        previous_mean = _window_mean(prefix, index - block_size - 1, block_size)
        local_drift = abs(recent_mean - previous_mean) / block_size
        ema = ema_alpha * local_drift + (1.0 - ema_alpha) * ema
        drift[index] = ema
    return drift


def run_fixed_window_baseline(
    values: np.ndarray,
    *,
    window: int,
    horizon_cap: np.ndarray | None = None,
) -> HorizonBaselineResult:
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    window_sizes = np.full(values.size, float(window))
    prefix = _prefix_sums(values)

    for index in range(values.size):
        effective_window = int(window)
        if horizon_cap is not None:
            effective_window = min(effective_window, int(round(horizon_cap[index])))
        effective_window = max(1, min(effective_window, index + 1))
        window_sizes[index] = float(effective_window)
        if index >= effective_window - 1:
            mean_value = _window_mean(prefix, index, effective_window)
            estimate[index] = mean_value
            sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)

    return HorizonBaselineResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=window_sizes,
        drift_estimate=np.zeros(values.size, dtype=float),
    )


def run_ewma_baseline(
    values: np.ndarray,
    *,
    alpha: float,
    horizon_cap: np.ndarray | None = None,
) -> HorizonBaselineResult:
    base_alpha = min(max(alpha, 1e-6), 1.0)
    base_width = float((2.0 - base_alpha) / base_alpha)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    window_sizes = np.full(values.size, base_width)
    if values.size == 0:
        return HorizonBaselineResult(
            sigma=sigma,
            estimate=estimate,
            window_sizes=window_sizes,
            drift_estimate=np.zeros(0, dtype=float),
        )

    estimate[0] = float(values[0])
    sigma[0] = 1.0
    for index in range(1, values.size):
        effective_width = base_width
        if horizon_cap is not None:
            effective_width = min(effective_width, float(horizon_cap[index]))
        effective_width = max(1.0, min(effective_width, float(index + 1)))
        alpha_t = min(1.0, 2.0 / (effective_width + 1.0))
        estimate[index] = (
            alpha_t * float(values[index]) + (1.0 - alpha_t) * estimate[index - 1]
        )
        sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - estimate[index]) ** 2)
        window_sizes[index] = effective_width

    return HorizonBaselineResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=window_sizes,
        drift_estimate=np.zeros(values.size, dtype=float),
    )


def compute_umr_regulator(
    values: np.ndarray,
    *,
    block_size: int,
    ema_alpha: float,
    baseline_window: int,
    prefix_length: int,
    scale: float,
    min_window: int,
    max_window: int,
) -> UMRRegulatorResult:
    drift = _local_drift_signal(values, block_size=block_size, ema_alpha=ema_alpha)
    calibration_prefix = max(1, min(prefix_length, values.size))
    prefix = values[:calibration_prefix]
    center = float(np.mean(prefix)) if prefix.size else 0.0
    ck = float(
        np.mean(np.abs(prefix - center)) * np.sqrt(max(baseline_window, 1)) * scale
    )
    ck = max(ck, 1e-9)
    safe_drift = np.maximum(drift, 1e-9)
    windows = np.clip((ck / safe_drift) ** (2.0 / 3.0), min_window, max_window)
    return UMRRegulatorResult(window_sizes=windows.astype(float), drift_estimate=drift)


def run_window_dilemma_baseline(
    values: np.ndarray,
    *,
    candidate_windows: Sequence[int],
    block_size: int,
    ema_alpha: float,
    baseline_window: int,
    prefix_length: int,
    low_quantile: float,
    high_quantile: float,
    horizon_cap: np.ndarray | None = None,
) -> HorizonBaselineResult:
    if len(candidate_windows) != 3:
        raise ValueError("candidate_windows must contain exactly three windows")

    windows_sorted = np.asarray(
        sorted(int(window) for window in candidate_windows), dtype=int
    )
    short_window, medium_window, long_window = [int(value) for value in windows_sorted]
    drift = _local_drift_signal(values, block_size=block_size, ema_alpha=ema_alpha)

    calibration = drift[2 * block_size : min(prefix_length, values.size)]
    calibration = calibration[calibration > 0]
    if calibration.size == 0:
        low_threshold = 0.0
        high_threshold = 0.0
    else:
        low_threshold = float(np.quantile(calibration, low_quantile))
        high_threshold = float(np.quantile(calibration, high_quantile))

    prefix = _prefix_sums(values)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    window_sizes = np.full(values.size, float(baseline_window))

    for index in range(values.size):
        current_drift = drift[index]
        if current_drift <= low_threshold:
            window = long_window
        elif current_drift >= high_threshold:
            window = short_window
        else:
            window = medium_window
        if horizon_cap is not None:
            window = min(window, int(round(horizon_cap[index])))
        window = max(1, min(window, index + 1))
        window_sizes[index] = float(window)
        if index >= window - 1:
            mean_value = _window_mean(prefix, index, window)
            estimate[index] = mean_value
            sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - mean_value) ** 2)

    return HorizonBaselineResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=window_sizes,
        drift_estimate=drift,
    )


def run_melo_style_baseline(
    values: np.ndarray,
    *,
    expert_windows: Sequence[int],
    learning_rate: float,
    discount: float,
    horizon_cap: np.ndarray | None = None,
) -> HorizonBaselineResult:
    windows = np.asarray(sorted({int(window) for window in expert_windows}), dtype=int)
    if windows.size == 0:
        raise ValueError("expert_windows must not be empty")

    prefix = _prefix_sums(values)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, np.nan)
    window_sizes = np.full(values.size, np.nan)
    effective_losses = np.zeros(windows.size, dtype=float)
    drift_estimate = np.zeros(values.size, dtype=float)

    for index in range(values.size):
        valid = windows <= index + 1
        if horizon_cap is not None:
            valid &= windows <= int(round(horizon_cap[index]))
        if not np.any(valid):
            continue

        predictions = np.asarray(
            [_window_mean(prefix, index, int(window)) for window in windows[valid]],
            dtype=float,
        )
        valid_windows = windows[valid].astype(float)

        scaled_losses = -learning_rate * effective_losses[valid]
        scaled_losses -= float(np.max(scaled_losses))
        weights = np.exp(scaled_losses)
        weights /= float(np.sum(weights))

        combined_estimate = float(np.dot(weights, predictions))
        estimate[index] = combined_estimate
        window_sizes[index] = float(np.dot(weights, valid_windows))
        sigma[index] = 1.0 / (1.0 + 0.5 * (values[index] - combined_estimate) ** 2)
        drift_estimate[index] = float(
            np.sqrt(np.dot(weights, (predictions - combined_estimate) ** 2))
        )

        prediction_losses = (values[index] - predictions) ** 2
        effective_losses[valid] = discount * effective_losses[valid] + prediction_losses

    return HorizonBaselineResult(
        sigma=sigma,
        estimate=estimate,
        window_sizes=window_sizes,
        drift_estimate=drift_estimate,
    )
