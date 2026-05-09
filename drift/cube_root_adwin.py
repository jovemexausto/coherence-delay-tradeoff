from __future__ import annotations

from collections import deque

from river import drift as river_drift


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


def calibrate_Ck(stream_prefix: list[float], n_calib: int | None = None) -> float:
    """Calibrate the cube-root constant from a stationary prefix."""

    n = n_calib or len(stream_prefix)
    data = stream_prefix[:n]
    if not data:
        return 1.0
    mean = sum(data) / len(data)
    var = sum((x - mean) ** 2 for x in data) / len(data)
    std = var**0.5
    return max(std * (len(data) ** 0.5), 1e-6)


class UMR:
    """ADWIN plus a cube-root cap on adaptive memory."""

    def __init__(
        self,
        *,
        delta: float = 0.002,
        Ck: float = 1.0,
        drift_window: int = 50,
        ema_alpha: float = 0.05,
        n_min: int = 10,
        n_max: int = 500,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        self.delta = float(delta)
        self.Ck = float(Ck)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self._clock = int(clock)
        self._max_buckets = int(max_buckets)
        self._min_window_length = int(min_window_length)
        self._grace_period = int(grace_period)
        self._adwin = self._new_adwin()
        self._drift_est = OnlineDriftEstimator(
            window=drift_window,
            ema_alpha=ema_alpha,
        )
        self._buffer: deque[float] = deque(maxlen=n_max)
        self._n_star = n_max
        self._cap_triggered = False

    @property
    def width(self) -> int:
        return int(self._adwin.width)

    @property
    def mean(self) -> float:
        return float(self._adwin.estimation)

    @property
    def drift_detected(self) -> bool:
        return bool(self._adwin.drift_detected)

    @property
    def cap_triggered(self) -> bool:
        return self._cap_triggered

    @property
    def n_star(self) -> int:
        return self._n_star

    @property
    def delta_est(self) -> float:
        return self._drift_est.delta

    def _compute_n_star(self) -> int:
        raw = (self.Ck / self._drift_est.delta) ** (2.0 / 3.0)
        return int(min(max(raw, self.n_min), self.n_max))

    def _new_adwin(self) -> river_drift.ADWIN:
        return river_drift.ADWIN(
            delta=self.delta,
            clock=self._clock,
            max_buckets=self._max_buckets,
            min_window_length=self._min_window_length,
            grace_period=self._grace_period,
        )

    def _rebuild_from_recent(self) -> None:
        recent = list(self._buffer)[-self._n_star :]
        detected = self._adwin.drift_detected
        self._adwin = self._new_adwin()
        for value in recent:
            self._adwin.update(value)
        self._adwin._drift_detected = detected

    def update(self, x: float) -> bool:
        self._cap_triggered = False
        x = float(x)
        self._drift_est.update(x)
        self._n_star = self._compute_n_star()
        self._buffer.append(x)
        self._adwin.update(x)

        if self.width > self._n_star:
            self._cap_triggered = True
            self._rebuild_from_recent()

        return self.drift_detected
