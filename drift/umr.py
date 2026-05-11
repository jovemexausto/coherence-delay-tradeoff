from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass(slots=True)
class DriftCalibrationResult:
    window: int
    ema_alpha: float
    score: float
    prediction_mse: float
    smoothness_penalty: float
    residual_std: float


class OnlineDriftEstimator:
    """Estimate a smoothed local drift scale from adjacent block shifts.

    The estimator is intentionally backend-agnostic: it only produces a local
    drift magnitude, which can then be converted into a horizon cap for any
    memory-bearing backend.
    """

    def __init__(self, window: int = 50, ema_alpha: float = 0.05, floor: float = 1e-6):
        self.window = int(window)
        self.ema_alpha = float(ema_alpha)
        self.floor = float(floor)
        self._buffer: deque[float] = deque(maxlen=2 * self.window)
        self._delta_ema: float = self.floor

    def update(self, x: float) -> float:
        self._buffer.append(float(x))
        if len(self._buffer) >= 2 * self.window:
            values = list(self._buffer)
            previous = values[: self.window]
            recent = values[self.window :]
            raw_delta = abs(sum(recent) / self.window - sum(previous) / self.window)
            raw_delta /= self.window
            self._delta_ema = (
                self.ema_alpha * raw_delta + (1.0 - self.ema_alpha) * self._delta_ema
            )
        return self.delta

    @property
    def delta(self) -> float:
        return max(self._delta_ema, self.floor)


def block_drift_series(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    prefix = np.zeros(values.size + 1, dtype=float)
    prefix[1:] = np.cumsum(values, dtype=float)
    drift = np.full(values.size, np.nan, dtype=float)
    for index in range(2 * window, values.size):
        right = index
        prev_left = right - 2 * window
        prev_right = right - window
        recent_mean = float((prefix[right] - prefix[prev_right]) / window)
        previous_mean = float((prefix[prev_right] - prefix[prev_left]) / window)
        drift[index] = abs(recent_mean - previous_mean) / window
    return drift


def score_ema_drift_proxy(
    values: np.ndarray,
    *,
    window: int,
    ema_alpha: float,
    smoothness_weight: float = 0.1,
    prefix_length: int | None = None,
) -> DriftCalibrationResult:
    prefix_length = values.size if prefix_length is None else int(prefix_length)
    prefix_length = max(0, min(prefix_length, values.size))
    series = block_drift_series(values[:prefix_length], window=window)
    valid = np.isfinite(series)
    if not np.any(valid):
        return DriftCalibrationResult(
            window=int(window),
            ema_alpha=float(ema_alpha),
            score=float("inf"),
            prediction_mse=float("inf"),
            smoothness_penalty=float("inf"),
            residual_std=float("inf"),
        )

    valid_series = series[valid]
    ema = float(valid_series[0])
    prev_ema = ema
    squared_residuals: list[float] = []
    squared_jumps: list[float] = []
    residuals: list[float] = []

    alpha = float(min(max(ema_alpha, 1e-6), 1.0))
    for delta in valid_series[1:]:
        residual = float(delta) - ema
        residuals.append(residual)
        squared_residuals.append(residual**2)
        squared_jumps.append((ema - prev_ema) ** 2)
        prev_ema = ema
        ema = alpha * float(delta) + (1.0 - alpha) * ema

    prediction_mse = float(np.mean(squared_residuals)) if squared_residuals else 0.0
    smoothness_penalty = float(np.mean(squared_jumps)) if squared_jumps else 0.0
    score = prediction_mse + smoothness_weight * smoothness_penalty
    residual_std = float(np.std(np.asarray(residuals, dtype=float)))
    return DriftCalibrationResult(
        window=int(window),
        ema_alpha=alpha,
        score=score,
        prediction_mse=prediction_mse,
        smoothness_penalty=smoothness_penalty,
        residual_std=residual_std,
    )


def select_drift_proxy_calibration(
    values: np.ndarray,
    *,
    windows: tuple[int, ...],
    alphas: tuple[float, ...],
    prefix_length: int,
    smoothness_weight: float = 0.1,
) -> DriftCalibrationResult:
    best: DriftCalibrationResult | None = None
    for window in windows:
        for alpha in alphas:
            candidate = score_ema_drift_proxy(
                values,
                window=window,
                ema_alpha=alpha,
                smoothness_weight=smoothness_weight,
                prefix_length=prefix_length,
            )
            if best is None or candidate.score < best.score:
                best = candidate
    if best is None:
        raise ValueError("windows and alphas must not be empty")
    return best


def calibrate_umr_constant(
    stream_prefix: list[float], n_calib: int | None = None
) -> float:
    """Calibrate the useful-memory constant from a stationary prefix."""

    n = n_calib or len(stream_prefix)
    data = stream_prefix[:n]
    if not data:
        return 1.0
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / len(data)
    std = var**0.5
    return max(std * (len(data) ** 0.5), 1e-6)


class UsefulMemoryRegulator:
    """Backend-agnostic temporal-validity cap driven by local drift.

    UMR does not implement a detector or predictor itself. It maintains a
    smoothed drift proxy and converts it into a runtime horizon bound that can
    be applied to fixed windows, exponential smoothers, ensembles, or detector
    wrappers with an internal width.
    """

    def __init__(
        self,
        *,
        Ck: float = 1.0,
        drift_window: int = 50,
        ema_alpha: float = 0.05,
        n_min: int = 10,
        n_max: int = 500,
    ) -> None:
        self.Ck = float(Ck)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self._drift_est = OnlineDriftEstimator(
            window=drift_window,
            ema_alpha=ema_alpha,
        )
        self._n_star = self.n_max

    @property
    def n_star(self) -> int:
        return self._n_star

    @property
    def delta_est(self) -> float:
        return self._drift_est.delta

    def observe(self, x: float) -> int:
        """Update the drift proxy with a new observation and return the cap."""
        self._drift_est.update(float(x))
        raw = (self.Ck / self._drift_est.delta) ** (2.0 / 3.0)
        self._n_star = int(min(max(raw, self.n_min), self.n_max))
        return self._n_star

    def limit(self, proposed_horizon: int | float) -> int:
        """Apply the current horizon cap to a backend-proposed memory scale."""
        return int(min(max(1, int(round(proposed_horizon))), self._n_star))
