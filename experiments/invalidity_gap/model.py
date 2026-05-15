from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river import drift as river_drift

from ..core.common import rolling_mean


@dataclass(slots=True)
class InvalidityGapConfig:
    seeds: tuple[int, ...] = tuple(range(12))
    steps: int = 3600
    warmup: int = 400
    phase_lengths: tuple[int, int, int] = (1000, 1400, 1200)
    low_drift: float = 0.00008
    high_drift: float = 0.0025
    observation_scale: float = 1.0
    process_scale: float = 0.0
    operating_window: int = 220
    detector_delta: float = 0.002
    detector_deltas: tuple[float, ...] = (0.0005, 0.001, 0.002, 0.004)
    n_min: int = 20
    n_max: int = 320
    Ck: float = 1.0
    reference_windows: tuple[int, ...] = tuple(range(20, 321, 20))
    rolling_window: int = 80
    persistence: int = 40


@dataclass(slots=True)
class InvalidityGapTrace:
    seed: int
    time: np.ndarray
    drift_path: np.ndarray
    latent_mean: np.ndarray
    observations: np.ndarray
    operating_estimate: np.ndarray
    oracle_estimate: np.ndarray
    operating_error: np.ndarray
    oracle_error: np.ndarray
    oracle_horizon: np.ndarray
    residual_stream: np.ndarray
    detector_events: np.ndarray
    t_valid: int | None
    t_detect: int | None
    invalidity_gap: int | None


@dataclass(slots=True)
class InvalidityGapSummary:
    detector_delta: float
    mean_t_valid: float
    mean_t_detect: float
    mean_gap: float
    std_gap: float
    positive_gap_rate: float
    detection_rate: float


@dataclass(slots=True)
class InvalidityGapResult:
    config: InvalidityGapConfig
    representative: InvalidityGapTrace
    traces: list[InvalidityGapTrace]
    summaries: list[InvalidityGapSummary]


def _simulate_stream(
    seed: int,
    config: InvalidityGapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p1, p2, p3 = config.phase_lengths
    drift_path = np.empty(config.steps, dtype=float)
    drift_path[:p1] = config.low_drift
    drift_path[p1 : p1 + p2] = np.linspace(
        config.low_drift,
        config.high_drift,
        p2,
        endpoint=True,
        dtype=float,
    )
    drift_path[p1 + p2 :] = config.high_drift

    latent_mean = np.zeros(config.steps, dtype=float)
    for step in range(1, config.steps):
        latent_mean[step] = (
            latent_mean[step - 1]
            + drift_path[step]
            + rng.normal(scale=config.process_scale)
        )
    observations = latent_mean + rng.normal(
        scale=config.observation_scale, size=config.steps
    )
    return drift_path, latent_mean, observations


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    estimate = np.zeros(values.size, dtype=float)
    for index in range(values.size):
        start = max(0, index - window + 1)
        estimate[index] = float(np.mean(values[start : index + 1]))
    return estimate


def _oracle_horizon(config: InvalidityGapConfig, drift_path: np.ndarray) -> np.ndarray:
    horizon = (config.Ck / np.maximum(drift_path, 1e-12)) ** (2.0 / 3.0)
    return np.clip(horizon, config.n_min, config.n_max)


def _variable_window_estimate(values: np.ndarray, windows: np.ndarray) -> np.ndarray:
    estimate = np.zeros(values.size, dtype=float)
    for index in range(values.size):
        window = max(1, min(int(round(float(windows[index]))), index + 1))
        start = max(0, index - window + 1)
        estimate[index] = float(np.mean(values[start : index + 1]))
    return estimate


def _first_persistent_true(
    mask: np.ndarray, persistence: int, warmup: int
) -> int | None:
    run = 0
    for index in range(max(warmup, 0), mask.size):
        run = run + 1 if mask[index] else 0
        if run >= persistence:
            return index - persistence + 1
    return None


def _first_detector_time(events: np.ndarray, start: int | None) -> int | None:
    if start is None:
        return None
    hits = np.flatnonzero(events[start:])
    return None if hits.size == 0 else int(start + hits[0])


def _run_single_trace(
    seed: int,
    config: InvalidityGapConfig,
    detector_delta: float,
) -> InvalidityGapTrace:
    drift_path, latent_mean, observations = _simulate_stream(seed, config)
    oracle_horizon = _oracle_horizon(config, drift_path)
    operating_estimate = _moving_average(observations, config.operating_window)
    oracle_estimate = _variable_window_estimate(observations, oracle_horizon)
    operating_error = np.abs(latent_mean - operating_estimate)
    oracle_error = np.abs(latent_mean - oracle_estimate)
    residual_stream = np.abs(observations - operating_estimate)

    detector = river_drift.ADWIN(delta=detector_delta)
    detector_events = np.zeros(config.steps, dtype=bool)
    for index, value in enumerate(observations):
        detector.update(float(value))
        detector_events[index] = bool(detector.drift_detected)

    stale_mask = config.operating_window > oracle_horizon
    t_valid = _first_persistent_true(stale_mask, config.persistence, config.warmup)
    t_detect = _first_detector_time(detector_events, t_valid)
    invalidity_gap = None if t_valid is None or t_detect is None else t_detect - t_valid

    return InvalidityGapTrace(
        seed=seed,
        time=np.arange(config.steps),
        drift_path=drift_path,
        latent_mean=latent_mean,
        observations=observations,
        operating_estimate=operating_estimate,
        oracle_estimate=oracle_estimate,
        operating_error=operating_error,
        oracle_error=oracle_error,
        oracle_horizon=oracle_horizon,
        residual_stream=residual_stream,
        detector_events=detector_events,
        t_valid=t_valid,
        t_detect=t_detect,
        invalidity_gap=invalidity_gap,
    )


def _summarize(
    traces: list[InvalidityGapTrace],
    detector_delta: float,
) -> InvalidityGapSummary:
    t_valid = [trace.t_valid for trace in traces if trace.t_valid is not None]
    t_detect = [trace.t_detect for trace in traces if trace.t_detect is not None]
    gaps = [
        trace.invalidity_gap for trace in traces if trace.invalidity_gap is not None
    ]
    return InvalidityGapSummary(
        detector_delta=detector_delta,
        mean_t_valid=float(np.mean(t_valid)) if t_valid else float("nan"),
        mean_t_detect=float(np.mean(t_detect)) if t_detect else float("nan"),
        mean_gap=float(np.mean(gaps)) if gaps else float("nan"),
        std_gap=float(np.std(gaps)) if gaps else float("nan"),
        positive_gap_rate=float(np.mean([gap > 0 for gap in gaps])) if gaps else 0.0,
        detection_rate=float(np.mean([trace.t_detect is not None for trace in traces])),
    )


def run_invalidity_gap_experiment(
    config: InvalidityGapConfig | None = None,
) -> InvalidityGapResult:
    cfg = config or InvalidityGapConfig()
    traces = [_run_single_trace(seed, cfg, cfg.detector_delta) for seed in cfg.seeds]
    summaries = []
    for delta in cfg.detector_deltas:
        delta_traces = [_run_single_trace(seed, cfg, delta) for seed in cfg.seeds]
        summaries.append(_summarize(delta_traces, delta))

    representative = max(
        traces,
        key=lambda trace: -1 if trace.invalidity_gap is None else trace.invalidity_gap,
    )
    return InvalidityGapResult(
        config=cfg,
        representative=representative,
        traces=traces,
        summaries=summaries,
    )


def rolling_error(
    trace: InvalidityGapTrace, window: int
) -> tuple[np.ndarray, np.ndarray]:
    return (
        rolling_mean(trace.operating_error, window),
        rolling_mean(trace.oracle_error, window),
    )
