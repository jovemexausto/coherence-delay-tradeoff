from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river import drift as river_drift

from drift.umr import calibrate_umr_constant

from ..core.detectors import run_umr_drift_detector


@dataclass(slots=True)
class UMRBenchmarkConfig:
    steps: int = 3000
    seeds: tuple[int, ...] = tuple(range(20))
    drift: float = 0.001
    drift_acceleration: float = 0.0
    piecewise_drifts: tuple[float, ...] = ()
    piecewise_lengths: tuple[int, ...] = ()
    observation_scale: float = 1.0
    process_scale: float = 0.0
    fixed_window: int = 100
    fixed_long_window: int = 500
    ewma_alpha: float = 0.05
    adwin_delta: float = 0.002
    drift_window: int = 50
    drift_ema_alpha: float = 0.05
    n_min: int = 10
    n_max: int = 500
    tail_fraction: float = 0.5
    Ck: float | None = None
    calibration_prefix: int = 500
    oracle_windows: tuple[int, ...] = (20, 30, 40, 50, 75, 100, 150, 200, 300, 500)


@dataclass(slots=True)
class MethodSeries:
    absolute_error_mean: np.ndarray
    absolute_error_std: np.ndarray
    memory_horizon_mean: np.ndarray
    memory_horizon_std: np.ndarray


@dataclass(slots=True)
class SeedTrace:
    latent_mean: np.ndarray
    observations: np.ndarray
    drift_path: np.ndarray
    fixed_estimate: np.ndarray
    fixed_long_estimate: np.ndarray
    ewma_estimate: np.ndarray
    adwin_estimate: np.ndarray
    cube_estimate: np.ndarray
    fixed_error: np.ndarray
    fixed_long_error: np.ndarray
    ewma_error: np.ndarray
    adwin_error: np.ndarray
    cube_error: np.ndarray
    fixed_width: np.ndarray
    fixed_long_width: np.ndarray
    ewma_width: np.ndarray
    adwin_width: np.ndarray
    cube_width: np.ndarray
    cube_n_star: np.ndarray
    adwin_drift_detected: list[int]
    cube_drift_detected: list[int]
    cube_cap_triggered: list[int]


@dataclass(slots=True)
class MethodSummary:
    tail_mae_mean: float
    tail_mae_std: float
    tail_rmse_mean: float
    tail_rmse_std: float
    tail_width_mean: float
    tail_width_std: float
    event_count_mean: float
    event_count_std: float
    cap_count_mean: float | None
    cap_count_std: float | None
    cap_only_count_mean: float | None
    cap_only_count_std: float | None
    first_cap_time_mean: float | None
    first_drift_time_mean: float | None
    cap_before_drift_delay_mean: float | None


@dataclass(slots=True)
class UMRBenchmarkResult:
    config: UMRBenchmarkConfig
    time: np.ndarray
    representative: SeedTrace
    traces: list[SeedTrace]
    series: dict[str, MethodSeries]
    summaries: dict[str, MethodSummary]


def _simulate_stream(
    *,
    seed: int,
    steps: int,
    drift: float,
    drift_acceleration: float,
    piecewise_drifts: tuple[float, ...],
    piecewise_lengths: tuple[int, ...],
    process_scale: float,
    observation_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = np.zeros(steps)
    obs = np.zeros(steps)
    if piecewise_drifts:
        if piecewise_lengths:
            if len(piecewise_lengths) != len(piecewise_drifts):
                raise ValueError("piecewise_lengths must match piecewise_drifts")
            phase_edges = [0]
            for length in piecewise_lengths:
                phase_edges.append(min(steps, phase_edges[-1] + max(0, int(length))))
            phase_edges[-1] = steps
            phase_edges = np.asarray(phase_edges, dtype=int)
        else:
            phase_edges = np.linspace(0, steps, len(piecewise_drifts) + 1, dtype=int)
        drift_path = np.zeros(steps)
        for phase, phase_drift in enumerate(piecewise_drifts):
            drift_path[phase_edges[phase] : phase_edges[phase + 1]] = phase_drift
    elif drift_acceleration != 0.0:
        progress = np.linspace(0.0, 1.0, steps)
        drift_path = drift + drift_acceleration * progress
    else:
        drift_path = np.full(steps, drift)
    for step in range(1, steps):
        latent[step] = (
            latent[step - 1] + drift_path[step] + rng.normal(scale=process_scale)
        )
    obs = latent + rng.normal(scale=observation_scale, size=steps)
    return latent, obs, drift_path


def _fixed_window_estimate(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, min(int(window), values.size))
    estimate = np.zeros(values.size)
    width = np.zeros(values.size)
    for index in range(values.size):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        estimate[index] = float(np.mean(chunk))
        width[index] = float(chunk.size)
    return estimate, width


def _ewma_estimate(values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = min(max(alpha, 1e-6), 1.0)
    estimate = np.zeros(values.size)
    width = np.full(values.size, (2.0 - alpha) / alpha)
    estimate[0] = float(values[0])
    for index in range(1, values.size):
        estimate[index] = (
            alpha * float(values[index]) + (1.0 - alpha) * estimate[index - 1]
        )
    return estimate, width


def _adwin_estimate(
    values: np.ndarray, *, delta: float
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    detector = river_drift.ADWIN(delta=delta)
    estimate = np.zeros(values.size)
    width = np.zeros(values.size)
    warnings: list[int] = []
    for index, value in enumerate(values):
        detector.update(float(value))
        estimate[index] = float(detector.estimation)
        width[index] = float(detector.width)
        if detector.drift_detected:
            warnings.append(index)
    return estimate, width, warnings


def _cube_root_estimate(
    values: np.ndarray,
    *,
    delta: float,
    Ck: float,
    drift_window: int,
    drift_ema_alpha: float,
    n_min: int,
    n_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    (
        warnings,
        width,
        n_star,
        _,
        caps,
        estimate,
    ) = run_umr_drift_detector(
        values,
        delta=delta,
        Ck=Ck,
        drift_window=drift_window,
        ema_alpha=drift_ema_alpha,
        n_min=n_min,
        n_max=n_max,
    )
    return estimate, width, n_star, warnings, caps


def _build_summary(
    traces: list[SeedTrace],
    method: str,
    tail_fraction: float,
) -> MethodSummary:
    first_cap: list[int | None]
    first_drift: list[int | None]
    if method == "fixed":
        errors = [trace.fixed_error for trace in traces]
        widths = [trace.fixed_width for trace in traces]
        events = [0 for _ in traces]
        caps = [0 for _ in traces]
        cap_only = [0 for _ in traces]
        first_cap = [None for _ in traces]
        first_drift = [None for _ in traces]
    elif method == "fixed_long":
        errors = [trace.fixed_long_error for trace in traces]
        widths = [trace.fixed_long_width for trace in traces]
        events = [0 for _ in traces]
        caps = [0 for _ in traces]
        cap_only = [0 for _ in traces]
        first_cap = [None for _ in traces]
        first_drift = [None for _ in traces]
    elif method == "ewma":
        errors = [trace.ewma_error for trace in traces]
        widths = [trace.ewma_width for trace in traces]
        events = [0 for _ in traces]
        caps = [0 for _ in traces]
        cap_only = [0 for _ in traces]
        first_cap = [None for _ in traces]
        first_drift = [None for _ in traces]
    elif method == "adwin":
        errors = [trace.adwin_error for trace in traces]
        widths = [trace.adwin_width for trace in traces]
        events = [len(trace.adwin_drift_detected) for trace in traces]
        caps = [0 for _ in traces]
        cap_only = [0 for _ in traces]
        first_cap = [None for _ in traces]
        first_drift = [
            trace.adwin_drift_detected[0] if trace.adwin_drift_detected else None
            for trace in traces
        ]
    elif method == "cube":
        errors = [trace.cube_error for trace in traces]
        widths = [trace.cube_width for trace in traces]
        events = [len(trace.cube_drift_detected) for trace in traces]
        caps = [len(trace.cube_cap_triggered) for trace in traces]
        cap_only = [
            len(set(trace.cube_cap_triggered) - set(trace.cube_drift_detected))
            for trace in traces
        ]
        first_cap = [
            trace.cube_cap_triggered[0] if trace.cube_cap_triggered else None
            for trace in traces
        ]
        first_drift = [
            trace.cube_drift_detected[0] if trace.cube_drift_detected else None
            for trace in traces
        ]
    else:
        raise ValueError(f"Unknown method: {method}")

    tail_mae: list[float] = []
    tail_rmse: list[float] = []
    tail_width: list[float] = []
    for error, width in zip(errors, widths, strict=True):
        tail_start = int(error.size * (1.0 - tail_fraction))
        tail_error = error[tail_start:]
        tail_window = width[tail_start:]
        tail_mae.append(float(np.mean(tail_error)))
        tail_rmse.append(float(np.sqrt(np.mean(tail_error**2))))
        tail_width.append(float(np.mean(tail_window)))

    def _maybe_mean(values: list[int | None]) -> float | None:
        valid = [float(v) for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    def _maybe_std(values: list[int | None]) -> float | None:
        valid = [float(v) for v in values if v is not None]
        return float(np.std(valid)) if valid else None

    cap_before_drift_delay = None
    if method == "cube":
        delays = [
            float(d - c)
            for c, d in zip(first_cap, first_drift, strict=True)
            if c is not None and d is not None and d >= c
        ]
        if delays:
            cap_before_drift_delay = float(np.mean(delays))

    return MethodSummary(
        tail_mae_mean=float(np.mean(tail_mae)),
        tail_mae_std=float(np.std(tail_mae)),
        tail_rmse_mean=float(np.mean(tail_rmse)),
        tail_rmse_std=float(np.std(tail_rmse)),
        tail_width_mean=float(np.mean(tail_width)),
        tail_width_std=float(np.std(tail_width)),
        event_count_mean=float(np.mean(events)),
        event_count_std=float(np.std(events)),
        cap_count_mean=float(np.mean(caps)) if method == "cube" else 0.0,
        cap_count_std=float(np.std(caps)) if method == "cube" else 0.0,
        cap_only_count_mean=float(np.mean(cap_only)) if method == "cube" else None,
        cap_only_count_std=float(np.std(cap_only)) if method == "cube" else None,
        first_cap_time_mean=_maybe_mean(first_cap),
        first_drift_time_mean=_maybe_mean(first_drift),
        cap_before_drift_delay_mean=cap_before_drift_delay,
    )


def run_benchmark(
    config: UMRBenchmarkConfig | None = None,
) -> UMRBenchmarkResult:
    config = config or UMRBenchmarkConfig()
    time = np.arange(config.steps)
    traces: list[SeedTrace] = []

    fixed_errors: list[np.ndarray] = []
    fixed_long_errors: list[np.ndarray] = []
    ewma_errors: list[np.ndarray] = []
    adwin_errors: list[np.ndarray] = []
    cube_errors: list[np.ndarray] = []
    fixed_widths: list[np.ndarray] = []
    fixed_long_widths: list[np.ndarray] = []
    ewma_widths: list[np.ndarray] = []
    adwin_widths: list[np.ndarray] = []
    cube_widths: list[np.ndarray] = []
    cube_n_stars: list[np.ndarray] = []

    for seed in config.seeds:
        latent, obs, drift_path = _simulate_stream(
            seed=seed,
            steps=config.steps,
            drift=config.drift,
            drift_acceleration=config.drift_acceleration,
            piecewise_drifts=config.piecewise_drifts,
            piecewise_lengths=config.piecewise_lengths,
            process_scale=config.process_scale,
            observation_scale=config.observation_scale,
        )
        Ck = (
            float(config.Ck)
            if config.Ck is not None
            else calibrate_umr_constant(list(obs[: config.calibration_prefix]))
        )
        fixed_estimate, fixed_width = _fixed_window_estimate(obs, config.fixed_window)
        fixed_long_estimate, fixed_long_width = _fixed_window_estimate(
            obs,
            config.fixed_long_window,
        )
        ewma_estimate, ewma_width = _ewma_estimate(obs, config.ewma_alpha)
        adwin_estimate, adwin_width, adwin_drift_detected = _adwin_estimate(
            obs,
            delta=config.adwin_delta,
        )
        (
            cube_estimate,
            cube_width,
            cube_n_star,
            cube_drift_detected,
            cube_cap_triggered,
        ) = _cube_root_estimate(
            obs,
            delta=config.adwin_delta,
            Ck=Ck,
            drift_window=config.drift_window,
            drift_ema_alpha=config.drift_ema_alpha,
            n_min=config.n_min,
            n_max=config.n_max,
        )

        fixed_error = np.abs(latent - fixed_estimate)
        fixed_long_error = np.abs(latent - fixed_long_estimate)
        ewma_error = np.abs(latent - ewma_estimate)
        adwin_error = np.abs(latent - adwin_estimate)
        cube_error = np.abs(latent - cube_estimate)

        traces.append(
            SeedTrace(
                latent_mean=latent,
                observations=obs,
                drift_path=drift_path,
                fixed_estimate=fixed_estimate,
                fixed_long_estimate=fixed_long_estimate,
                ewma_estimate=ewma_estimate,
                adwin_estimate=adwin_estimate,
                cube_estimate=cube_estimate,
                fixed_error=fixed_error,
                fixed_long_error=fixed_long_error,
                ewma_error=ewma_error,
                adwin_error=adwin_error,
                cube_error=cube_error,
                fixed_width=fixed_width,
                fixed_long_width=fixed_long_width,
                ewma_width=ewma_width,
                adwin_width=adwin_width,
                cube_width=cube_width,
                cube_n_star=cube_n_star,
                adwin_drift_detected=adwin_drift_detected,
                cube_drift_detected=cube_drift_detected,
                cube_cap_triggered=cube_cap_triggered,
            )
        )

        fixed_errors.append(fixed_error)
        fixed_widths.append(fixed_width)
        fixed_long_errors.append(fixed_long_error)
        fixed_long_widths.append(fixed_long_width)
        ewma_errors.append(ewma_error)
        adwin_errors.append(adwin_error)
        cube_errors.append(cube_error)
        ewma_widths.append(ewma_width)
        adwin_widths.append(adwin_width)
        cube_widths.append(cube_width)
        cube_n_stars.append(cube_n_star)

    series = {
        "fixed": MethodSeries(
            absolute_error_mean=np.mean(np.stack(fixed_errors), axis=0),
            absolute_error_std=np.std(np.stack(fixed_errors), axis=0),
            memory_horizon_mean=np.mean(np.stack(fixed_widths), axis=0),
            memory_horizon_std=np.std(np.stack(fixed_widths), axis=0),
        ),
        "fixed_long": MethodSeries(
            absolute_error_mean=np.mean(np.stack(fixed_long_errors), axis=0),
            absolute_error_std=np.std(np.stack(fixed_long_errors), axis=0),
            memory_horizon_mean=np.mean(np.stack(fixed_long_widths), axis=0),
            memory_horizon_std=np.std(np.stack(fixed_long_widths), axis=0),
        ),
        "ewma": MethodSeries(
            absolute_error_mean=np.mean(np.stack(ewma_errors), axis=0),
            absolute_error_std=np.std(np.stack(ewma_errors), axis=0),
            memory_horizon_mean=np.mean(np.stack(ewma_widths), axis=0),
            memory_horizon_std=np.std(np.stack(ewma_widths), axis=0),
        ),
        "adwin": MethodSeries(
            absolute_error_mean=np.mean(np.stack(adwin_errors), axis=0),
            absolute_error_std=np.std(np.stack(adwin_errors), axis=0),
            memory_horizon_mean=np.mean(np.stack(adwin_widths), axis=0),
            memory_horizon_std=np.std(np.stack(adwin_widths), axis=0),
        ),
        "cube": MethodSeries(
            absolute_error_mean=np.mean(np.stack(cube_errors), axis=0),
            absolute_error_std=np.std(np.stack(cube_errors), axis=0),
            memory_horizon_mean=np.mean(np.stack(cube_n_stars), axis=0),
            memory_horizon_std=np.std(np.stack(cube_n_stars), axis=0),
        ),
    }

    summaries = {
        "fixed": _build_summary(traces, "fixed", config.tail_fraction),
        "fixed_long": _build_summary(traces, "fixed_long", config.tail_fraction),
        "ewma": _build_summary(traces, "ewma", config.tail_fraction),
        "adwin": _build_summary(traces, "adwin", config.tail_fraction),
        "cube": _build_summary(traces, "cube", config.tail_fraction),
    }

    return UMRBenchmarkResult(
        config=config,
        time=time,
        representative=traces[0],
        traces=traces,
        series=series,
        summaries=summaries,
    )
