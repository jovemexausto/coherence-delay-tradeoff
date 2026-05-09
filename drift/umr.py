from __future__ import annotations

from collections import deque


class OnlineDriftEstimator:
    """Estimate local drift magnitude with an EMA of adjacent block shifts."""

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
    """Backend-agnostic horizon controller driven by local drift."""

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
        self._drift_est.update(float(x))
        raw = (self.Ck / self._drift_est.delta) ** (2.0 / 3.0)
        self._n_star = int(min(max(raw, self.n_min), self.n_max))
        return self._n_star

    def limit(self, proposed_horizon: int | float) -> int:
        return int(min(max(1, int(round(proposed_horizon))), self._n_star))
