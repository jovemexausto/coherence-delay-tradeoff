from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.io import arff


AIRLINES_DATA_URL = "https://www.openml.org/data/v1/download/21854421/airlines.arff"


@dataclass(slots=True)
class AirlinesConfig:
    data_url: str = AIRLINES_DATA_URL
    cache_path: Path = Path("artifacts/airlines/airlines.arff")
    hash_dim: int = 128
    learning_rate: float = 0.08
    l2: float = 1e-4
    calibration_samples: int = 5000
    horizon_windows: tuple[int, ...] = (10, 25, 50, 100, 200, 400)
    cap_block_size: int = 200
    cap_ema_alpha: float = 0.05
    cap_baseline_window: int = 500
    cap_prefix_length: int = 5000
    cap_scale: float = 1.25
    cap_min_window: int = 10
    cap_max_window: int = 400
    ensemble_temperature: float = 0.35


@dataclass(slots=True)
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return max(self.m2 / (self.count - 1), 1e-6)

    def transform(self, value: float) -> float:
        if self.count < 2:
            return 0.0
        return (value - self.mean) / np.sqrt(self.variance)


@dataclass(slots=True)
class OnlineLogisticModel:
    dimension: int
    learning_rate: float
    l2: float
    weights: np.ndarray
    bias: float = 0.0

    @classmethod
    def create(
        cls, dimension: int, learning_rate: float, l2: float
    ) -> "OnlineLogisticModel":
        return cls(
            dimension=dimension,
            learning_rate=learning_rate,
            l2=l2,
            weights=np.zeros(dimension, dtype=float),
            bias=0.0,
        )

    @staticmethod
    def _sigmoid(score: float) -> float:
        if score >= 0.0:
            z = np.exp(-score)
            return float(1.0 / (1.0 + z))
        z = np.exp(score)
        return float(z / (1.0 + z))

    def predict_proba(self, features: np.ndarray) -> float:
        score = float(np.dot(self.weights, features) + self.bias)
        return self._sigmoid(score)

    def learn_one(self, features: np.ndarray, target: int) -> None:
        prediction = self.predict_proba(features)
        error = prediction - float(target)
        self.weights -= self.learning_rate * (error * features + self.l2 * self.weights)
        self.bias -= self.learning_rate * error


@dataclass(slots=True)
class AirlinesBenchmarkResult:
    config: AirlinesConfig
    frame: pd.DataFrame
    targets: np.ndarray
    base_probabilities: np.ndarray
    window_sizes: np.ndarray
    window_probabilities: np.ndarray
    calibration_slice: slice
    test_slice: slice
    selected_window_index: int
    selected_window: int
    window_losses: np.ndarray
    window_weights: np.ndarray
    drift_proxy: np.ndarray
    caps: np.ndarray
    single_probabilities: np.ndarray
    ensemble_probabilities: np.ndarray
    single_cap_probabilities: np.ndarray
    ensemble_cap_probabilities: np.ndarray


_NUMERIC_COLUMNS = ("Time", "Length")
_CATEGORICAL_COLUMNS = ("Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek")


def _stable_bucket(text: str, bucket_count: int) -> int:
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_count


def _download_airlines_data(config: AirlinesConfig) -> Path:
    cache_path = config.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path

    with urlopen(config.data_url, timeout=60) as response:
        cache_path.write_bytes(response.read())
    return cache_path


def _decode_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    frame = raw_frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(
                lambda value: (
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
            )
    return frame


def load_airlines_frame(config: AirlinesConfig | None = None) -> pd.DataFrame:
    config = config or AirlinesConfig()
    data_path = _download_airlines_data(config)
    with data_path.open("rb") as handle:
        raw_data, _ = arff.loadarff(StringIO(handle.read().decode("utf-8")))
    frame = pd.DataFrame(raw_data)
    frame = _decode_frame(frame)
    frame["Delay"] = frame["Delay"].astype(int)
    frame["Time"] = frame["Time"].astype(float)
    frame["Length"] = frame["Length"].astype(float)
    frame["DayOfWeek"] = frame["DayOfWeek"].astype(str)
    frame["Flight"] = frame["Flight"].astype(str)
    return frame


def _encode_row(
    row: pd.Series,
    time_stats: RunningStats,
    length_stats: RunningStats,
    config: AirlinesConfig,
) -> np.ndarray:
    features = np.zeros(config.hash_dim, dtype=float)
    features[0] = 1.0
    features[1] = time_stats.transform(float(row["Time"]))
    features[2] = length_stats.transform(float(row["Length"]))

    if config.hash_dim <= 3:
        return features

    bucket_count = config.hash_dim - 3
    for name in _CATEGORICAL_COLUMNS:
        bucket = 3 + _stable_bucket(f"{name}={row[name]}", bucket_count)
        features[bucket] += 1.0
    return features


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), signal.size))
    output = np.full(signal.size, np.nan, dtype=float)
    prefix = np.zeros(signal.size + 1, dtype=float)
    prefix[1:] = np.cumsum(signal, dtype=float)
    for index in range(window - 1, signal.size):
        output[index] = (prefix[index + 1] - prefix[index + 1 - window]) / window
    return output


def _ema(signal: np.ndarray, alpha: float) -> np.ndarray:
    ema = np.zeros(signal.size, dtype=float)
    value = 0.0
    for index, point in enumerate(signal):
        value = alpha * float(point) + (1.0 - alpha) * value
        ema[index] = value
    return ema


def _build_cap_series(
    drift_proxy: np.ndarray, config: AirlinesConfig
) -> tuple[np.ndarray, np.ndarray]:
    block_smoothed = _moving_average(drift_proxy, config.cap_block_size)
    block_smoothed = np.where(np.isfinite(block_smoothed), block_smoothed, drift_proxy)
    drift = _ema(block_smoothed, config.cap_ema_alpha)
    prefix_length = max(1, min(config.cap_prefix_length, drift.size))
    prefix = drift[:prefix_length]
    baseline_level = float(np.mean(prefix) + np.std(prefix)) if prefix.size else 0.0
    ck = float(baseline_level * max(config.cap_baseline_window, 1) * config.cap_scale)
    ck = max(ck, 1e-9)
    caps = np.clip(
        (ck / np.maximum(drift, 1e-9)) ** (2.0 / 3.0),
        float(config.cap_min_window),
        float(config.cap_max_window),
    )
    return caps.astype(float), drift


def _binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-9
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(
        -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    )


def _summed_weights(losses: np.ndarray, temperature: float) -> np.ndarray:
    scaled = -losses / max(temperature, 1e-9)
    scaled -= float(np.max(scaled))
    weights = np.exp(scaled)
    return weights / float(np.sum(weights))


def run_airlines_benchmark(
    config: AirlinesConfig | None = None,
) -> AirlinesBenchmarkResult:
    config = config or AirlinesConfig()
    frame = load_airlines_frame(config)
    targets = frame["Delay"].to_numpy(dtype=int)

    windows = np.asarray(
        sorted({int(window) for window in config.horizon_windows}), dtype=int
    )
    if windows.size == 0:
        raise ValueError("horizon_windows must not be empty")

    model = OnlineLogisticModel.create(config.hash_dim, config.learning_rate, config.l2)
    time_stats = RunningStats()
    length_stats = RunningStats()

    base_probabilities = np.full(targets.size, np.nan, dtype=float)
    drift_proxy = np.zeros(targets.size, dtype=float)
    encoded_previous = np.zeros(config.hash_dim, dtype=float)

    for index, (_, row) in enumerate(frame.iterrows()):
        encoded = _encode_row(row, time_stats, length_stats, config)
        base_probabilities[index] = model.predict_proba(encoded)
        if index > 0:
            drift_proxy[index] = float(np.mean(np.abs(encoded - encoded_previous)))
        model.learn_one(encoded, int(targets[index]))
        time_stats.update(float(row["Time"]))
        length_stats.update(float(row["Length"]))
        encoded_previous = encoded

    caps, drift_signal = _build_cap_series(drift_proxy, config)

    window_probabilities = np.vstack(
        [_moving_average(base_probabilities, int(window)) for window in windows]
    )
    warmup = int(windows.max())
    calibration_end = min(max(config.calibration_samples, warmup + 1), targets.size)
    calibration_slice = slice(warmup, calibration_end)
    test_slice = slice(calibration_end, targets.size)

    window_losses = np.full(windows.size, np.nan, dtype=float)
    for index, window_prob in enumerate(window_probabilities):
        valid = np.isfinite(window_prob[calibration_slice])
        if not np.any(valid):
            window_losses[index] = float("inf")
            continue
        window_losses[index] = _binary_log_loss(
            targets[calibration_slice][valid], window_prob[calibration_slice][valid]
        )

    selected_window_index = int(np.nanargmin(window_losses))
    selected_window = int(windows[selected_window_index])
    window_weights = _summed_weights(window_losses, config.ensemble_temperature)

    single_probabilities = window_probabilities[selected_window_index].copy()
    ensemble_probabilities = np.full(targets.size, np.nan, dtype=float)
    for index in range(targets.size):
        valid = np.isfinite(window_probabilities[:, index])
        if not np.any(valid):
            continue
        local_weights = window_weights[valid]
        local_weights = local_weights / float(np.sum(local_weights))
        ensemble_probabilities[index] = float(
            np.dot(local_weights, window_probabilities[:, index][valid])
        )

    single_cap_probabilities = np.full(targets.size, np.nan, dtype=float)
    ensemble_cap_probabilities = np.full(targets.size, np.nan, dtype=float)
    for index in range(targets.size):
        allowed = np.flatnonzero(windows <= int(round(caps[index])))
        if allowed.size == 0:
            allowed = np.array([0], dtype=int)
        capped_single_index = allowed[-1]
        single_cap_probabilities[index] = window_probabilities[
            capped_single_index, index
        ]
        local_weights = window_weights[allowed]
        local_weights = local_weights / float(np.sum(local_weights))
        ensemble_cap_probabilities[index] = float(
            np.dot(local_weights, window_probabilities[allowed, index])
        )

    return AirlinesBenchmarkResult(
        config=config,
        frame=frame,
        targets=targets,
        base_probabilities=base_probabilities,
        window_sizes=windows,
        window_probabilities=window_probabilities,
        calibration_slice=calibration_slice,
        test_slice=test_slice,
        selected_window_index=selected_window_index,
        selected_window=selected_window,
        window_losses=window_losses,
        window_weights=window_weights,
        drift_proxy=drift_signal,
        caps=caps,
        single_probabilities=single_probabilities,
        ensemble_probabilities=ensemble_probabilities,
        single_cap_probabilities=single_cap_probabilities,
        ensemble_cap_probabilities=ensemble_cap_probabilities,
    )
