from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _default_candidate_windows() -> tuple[int, ...]:
    values = np.unique(np.round(np.geomspace(8, 256, 18)).astype(int)).tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class OnlineAdaptationConfig:
    phase_lengths: tuple[int, ...] = (300, 300, 300, 300)
    holder_exponents: tuple[float, ...] = (1.0, 0.75, 0.5, 1.0)
    roughness_scales: tuple[float, ...] = (0.001, 0.08, 0.001, 0.08)
    phase_signs: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)
    observation_scale: float = 1.0
    candidate_windows: tuple[int, ...] = _default_candidate_windows()
    roughness_block_size: int = 12
    lag_multipliers: tuple[int, ...] = (1, 2, 4, 6)
    direct_radius_log2: float = 1.0
    structural_radius_log2: float = 1.5
    warmup: int = 96
    carrier_exponent: float = 0.5
    validation_tail: int = 160
    aggregated_history: int = 160
    activity_psi_exponent: float = 0.5
    activity_smoothing: float = 0.9
    activity_grow_index_step: int = 1
    switch_relative_margin: float = 0.0
    max_window_index_jump: int | None = None
    holder_clip: tuple[float, float] = (0.35, 1.05)
    roughness_floor: float = 1e-4
    roughness_cap: float = 0.25
    seed: int = 0


@dataclass(frozen=True, slots=True)
class RoughnessEstimate:
    holder_exponent: float
    roughness_scale: float
    plugin_window: int


@dataclass(frozen=True, slots=True)
class OnlineAdaptationResult:
    config: OnlineAdaptationConfig
    time: np.ndarray
    latent_mean: np.ndarray
    observations: np.ndarray
    oracle_window: np.ndarray
    plugin_window: np.ndarray
    adaptive_window: np.ndarray
    structural_window: np.ndarray
    activity_window: np.ndarray
    plugin_estimate: np.ndarray
    oracle_estimate: np.ndarray
    adaptive_estimate: np.ndarray
    structural_estimate: np.ndarray
    activity_estimate: np.ndarray
    best_static_window: int
    best_static_estimate: np.ndarray
    oracle_error: np.ndarray
    plugin_error: np.ndarray
    adaptive_error: np.ndarray
    structural_error: np.ndarray
    activity_error: np.ndarray
    best_static_error: np.ndarray
    mean_oracle_error: float
    mean_plugin_error: float
    mean_adaptive_error: float
    mean_structural_error: float
    mean_activity_error: float
    mean_best_static_error: float


def phase_profile(
    config: OnlineAdaptationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_steps = sum(config.phase_lengths)
    latent_mean = np.zeros(total_steps, dtype=float)
    holder = np.zeros(total_steps, dtype=float)
    roughness = np.zeros(total_steps, dtype=float)
    start = 0
    level = 0.0
    for length, H, zeta, sign in zip(
        config.phase_lengths,
        config.holder_exponents,
        config.roughness_scales,
        config.phase_signs,
        strict=True,
    ):
        local = np.arange(length, dtype=float)
        latent_mean[start : start + length] = level + sign * zeta * local**H
        holder[start : start + length] = H
        roughness[start : start + length] = zeta
        level = float(latent_mean[start + length - 1])
        start += length
    return latent_mean, holder, roughness


def _block_mean(values: np.ndarray, end_index: int, block_size: int) -> float:
    start = end_index - block_size + 1
    return float(np.mean(values[start : end_index + 1]))


def _noise_floor(values: np.ndarray, end_index: int, block_size: int) -> float:
    start = end_index - block_size + 1
    block = values[start : end_index + 1]
    even = block[::2]
    odd = block[1::2]
    if odd.size == 0:
        return 0.0
    return abs(float(np.mean(even) - np.mean(odd))) / math.sqrt(2.0)


def estimate_local_roughness(
    observations: np.ndarray,
    time_index: int,
    config: OnlineAdaptationConfig,
) -> RoughnessEstimate:
    block_size = config.roughness_block_size
    if time_index < block_size * (max(config.lag_multipliers) + 1):
        raise ValueError("time_index is too small for local roughness estimation")
    recent_mean = _block_mean(observations, time_index, block_size)
    noise_floor = _noise_floor(observations, time_index, block_size)
    log_lags: list[float] = []
    log_discrepancies: list[float] = []
    for multiplier in config.lag_multipliers:
        lag = multiplier * block_size
        past_end = time_index - lag
        past_mean = _block_mean(observations, past_end, block_size)
        discrepancy = max(abs(recent_mean - past_mean) - noise_floor, 1e-8)
        log_lags.append(math.log(float(lag)))
        log_discrepancies.append(math.log(discrepancy))
    slope, intercept = np.polyfit(
        np.asarray(log_lags), np.asarray(log_discrepancies), 1
    )
    holder = float(np.clip(slope, config.holder_clip[0], config.holder_clip[1]))
    roughness = float(
        np.clip(math.exp(intercept), config.roughness_floor, config.roughness_cap)
    )
    plugin = plugin_window_from_roughness(holder, roughness, config)
    return RoughnessEstimate(
        holder_exponent=holder, roughness_scale=roughness, plugin_window=plugin
    )


def estimate_aggregated_roughness(
    observations: np.ndarray,
    time_index: int,
    config: OnlineAdaptationConfig,
) -> RoughnessEstimate:
    block_size = config.roughness_block_size
    if time_index < block_size * (max(config.lag_multipliers) + 1):
        raise ValueError("time_index is too small for aggregated roughness estimation")
    noise_floor = _noise_floor(observations, time_index, block_size)
    log_lags: list[float] = []
    log_summaries: list[float] = []
    summaries: list[float] = []
    lags: list[float] = []
    history = max(config.aggregated_history, block_size)
    for multiplier in config.lag_multipliers:
        lag = multiplier * block_size
        start = max(block_size - 1 + lag, time_index - history + 1)
        discrepancies: list[float] = []
        for index in range(start, time_index + 1):
            recent_mean = _block_mean(observations, index, block_size)
            past_mean = _block_mean(observations, index - lag, block_size)
            discrepancies.append(max(abs(recent_mean - past_mean) - noise_floor, 1e-8))
        summary = float(np.mean(discrepancies))
        summaries.append(summary)
        lags.append(float(lag))
        log_lags.append(math.log(float(lag)))
        log_summaries.append(math.log(summary))
    slope, _ = np.polyfit(np.asarray(log_lags), np.asarray(log_summaries), 1)
    holder = float(np.clip(slope, config.holder_clip[0], config.holder_clip[1]))
    roughness = float(
        np.clip(
            np.median(np.asarray(summaries) / (np.asarray(lags) ** holder)),
            config.roughness_floor,
            config.roughness_cap,
        )
    )
    plugin = plugin_window_from_roughness(holder, roughness, config)
    return RoughnessEstimate(
        holder_exponent=holder, roughness_scale=roughness, plugin_window=plugin
    )


def plugin_window_from_roughness(
    holder_exponent: float,
    roughness_scale: float,
    config: OnlineAdaptationConfig,
) -> int:
    carrier_constant = config.observation_scale * math.sqrt(2.0 / math.pi)
    exponent = 1.0 / (config.carrier_exponent + holder_exponent)
    raw = (carrier_constant / max(roughness_scale, 1e-12)) ** exponent
    return int(
        np.clip(
            round(raw), min(config.candidate_windows), max(config.candidate_windows)
        )
    )


def oracle_window_from_truth(
    holder_exponent: float,
    roughness_scale: float,
    config: OnlineAdaptationConfig,
) -> int:
    return plugin_window_from_roughness(holder_exponent, roughness_scale, config)


def oracle_window_from_latent(
    latent_mean: np.ndarray,
    time_index: int,
    config: OnlineAdaptationConfig,
) -> int:
    carrier_constant = config.observation_scale * math.sqrt(2.0 / math.pi)
    best_window = min(config.candidate_windows)
    best_score = float("inf")
    for window in config.candidate_windows:
        usable = min(window, time_index + 1)
        target = float(np.mean(latent_mean[time_index - usable + 1 : time_index + 1]))
        score = carrier_constant / math.sqrt(float(usable)) + abs(
            float(latent_mean[time_index]) - target
        )
        if score < best_score:
            best_score = score
            best_window = usable
    return best_window


def _candidate_band(center: int, config: OnlineAdaptationConfig) -> tuple[int, ...]:
    lower = center / (2.0**config.direct_radius_log2)
    upper = center * (2.0**config.direct_radius_log2)
    return tuple(
        n
        for n in config.candidate_windows
        if lower <= n <= upper and n >= config.roughness_block_size
    )


def _moving_average(values: np.ndarray, end_index: int, window: int) -> float:
    usable = min(window, end_index + 1)
    start = end_index - usable + 1
    return float(np.mean(values[start : end_index + 1]))


def _direct_score(values: np.ndarray, end_index: int, window: int) -> float:
    window = min(window, end_index + 1)
    if window < 4:
        return float("inf")
    start = end_index - window + 1
    block = values[start : end_index + 1]
    odd = block[::2]
    even = block[1::2]
    if even.size == 0:
        return float("inf")
    carrier_proxy = abs(float(np.mean(odd) - np.mean(even))) / math.sqrt(2.0)
    half = max(window // 2, 2)
    short_mean = float(np.mean(values[end_index - half + 1 : end_index + 1]))
    full_mean = float(np.mean(block))
    staleness_proxy = abs(full_mean - short_mean)
    return carrier_proxy + staleness_proxy


def _validation_score(
    values: np.ndarray, end_index: int, window: int, tail: int
) -> float:
    start = max(window, end_index - tail + 1)
    errors: list[float] = []
    for index in range(start, end_index + 1):
        prediction = _moving_average(values, index - 1, window)
        errors.append(abs(float(values[index]) - prediction))
    return float(np.mean(errors)) if errors else float("inf")


def estimate_local_activity(
    observations: np.ndarray, time_index: int, config: OnlineAdaptationConfig
) -> float:
    block_size = config.roughness_block_size
    if time_index < 2 * block_size:
        return 0.0
    recent_mean = _block_mean(observations, time_index, block_size)
    previous_mean = _block_mean(observations, time_index - block_size, block_size)
    noise_floor = _noise_floor(observations, time_index, block_size)
    discrepancy = max(abs(recent_mean - previous_mean) - noise_floor, 0.0)
    return discrepancy / max(float(block_size) ** config.activity_psi_exponent, 1.0)


def activity_window_from_proxy(
    activity_proxy: float, config: OnlineAdaptationConfig
) -> int:
    carrier_constant = config.observation_scale * math.sqrt(2.0 / math.pi)
    return min(
        config.candidate_windows,
        key=lambda window: (
            carrier_constant / math.sqrt(window)
            + activity_proxy * (window**config.activity_psi_exponent)
        ),
    )


def horizon_window_from_activity(
    holder_exponent: float,
    activity_proxy: float,
    config: OnlineAdaptationConfig,
) -> int:
    carrier_constant = config.observation_scale * math.sqrt(2.0 / math.pi)
    return min(
        config.candidate_windows,
        key=lambda window: (
            carrier_constant / math.sqrt(window)
            + activity_proxy * (window**holder_exponent)
        ),
    )


def _project_window_to_grid(
    window: int, config: OnlineAdaptationConfig, end_index: int
) -> int:
    usable_max = end_index + 1
    feasible = tuple(
        candidate for candidate in config.candidate_windows if candidate <= usable_max
    )
    if not feasible:
        return min(config.candidate_windows)
    return min(feasible, key=lambda candidate: abs(candidate - window))


def _activity_window(
    values: np.ndarray,
    end_index: int,
    config: OnlineAdaptationConfig,
    previous_proxy: float,
    previous_window: int,
) -> tuple[float, int]:
    current_proxy = estimate_local_activity(values, end_index, config)
    activity_proxy = (
        config.activity_smoothing * previous_proxy
        + (1.0 - config.activity_smoothing) * current_proxy
    )
    target_window = _project_window_to_grid(
        activity_window_from_proxy(activity_proxy, config), config, end_index
    )
    previous_index = config.candidate_windows.index(previous_window)
    target_index = config.candidate_windows.index(target_window)
    if target_index <= previous_index:
        chosen_window = target_window
    else:
        step = min(config.activity_grow_index_step, target_index - previous_index)
        chosen_window = config.candidate_windows[previous_index + step]
    return activity_proxy, chosen_window


def _structural_window(
    values: np.ndarray,
    end_index: int,
    config: OnlineAdaptationConfig,
    previous_proxy: float,
    previous_window: int,
) -> tuple[RoughnessEstimate, float]:
    roughness = estimate_aggregated_roughness(values, end_index, config)
    current_proxy = estimate_local_activity(values, end_index, config)
    activity_proxy = (
        config.activity_smoothing * previous_proxy
        + (1.0 - config.activity_smoothing) * current_proxy
    )
    center_window = _project_window_to_grid(
        horizon_window_from_activity(
            roughness.holder_exponent, roughness.roughness_scale, config
        ),
        config,
        end_index,
    )
    lower = center_window / (2.0**config.structural_radius_log2)
    upper = center_window * (2.0**config.structural_radius_log2)
    candidate_windows = tuple(
        window
        for window in config.candidate_windows
        if lower <= window <= upper and window >= config.roughness_block_size
    )
    feasible_windows = tuple(
        window for window in candidate_windows if window <= end_index + 1
    )
    if not feasible_windows:
        feasible_windows = tuple(
            window for window in config.candidate_windows if window <= end_index + 1
        )
    scores = {
        window: _validation_score(values, end_index, window, config.validation_tail)
        for window in feasible_windows
    }
    best_window = min(scores, key=scores.get)
    if config.max_window_index_jump is not None:
        previous_index = config.candidate_windows.index(previous_window)
        best_index = config.candidate_windows.index(best_window)
        step = max(
            -config.max_window_index_jump,
            min(config.max_window_index_jump, best_index - previous_index),
        )
        best_window = config.candidate_windows[previous_index + step]
    best_window = _project_window_to_grid(best_window, config, end_index)
    return (
        RoughnessEstimate(
            holder_exponent=roughness.holder_exponent,
            roughness_scale=roughness.roughness_scale,
            plugin_window=best_window,
        ),
        activity_proxy,
    )


def _adaptive_window(
    values: np.ndarray,
    end_index: int,
    config: OnlineAdaptationConfig,
    previous_window: int,
) -> RoughnessEstimate:
    estimate = estimate_local_roughness(values, end_index, config)
    scores = {
        window: _validation_score(values, end_index, window, config.validation_tail)
        for window in config.candidate_windows
    }
    best_window = min(scores, key=scores.get)
    previous_score = scores.get(previous_window, float("inf"))
    if scores[best_window] > (1.0 - config.switch_relative_margin) * previous_score:
        best_window = previous_window
    elif config.max_window_index_jump is not None:
        prev_index = config.candidate_windows.index(previous_window)
        best_index = config.candidate_windows.index(best_window)
        step = max(
            -config.max_window_index_jump,
            min(config.max_window_index_jump, best_index - prev_index),
        )
        best_window = config.candidate_windows[prev_index + step]
    best_window = _project_window_to_grid(best_window, config, end_index)
    return RoughnessEstimate(
        holder_exponent=estimate.holder_exponent,
        roughness_scale=estimate.roughness_scale,
        plugin_window=best_window,
    )


def run_online_horizon_adaptation_experiment(
    config: OnlineAdaptationConfig | None = None,
) -> OnlineAdaptationResult:
    cfg = config or OnlineAdaptationConfig()
    latent_mean, holder_truth, roughness_truth = phase_profile(cfg)
    rng = np.random.default_rng(cfg.seed)
    observations = latent_mean + rng.normal(
        scale=cfg.observation_scale, size=latent_mean.size
    )
    time = np.arange(latent_mean.size, dtype=int)
    oracle_window = np.full(latent_mean.size, min(cfg.candidate_windows), dtype=int)
    plugin_window = np.full(latent_mean.size, min(cfg.candidate_windows), dtype=int)
    adaptive_window = np.full(latent_mean.size, min(cfg.candidate_windows), dtype=int)
    structural_window = np.full(latent_mean.size, min(cfg.candidate_windows), dtype=int)
    activity_window = np.full(latent_mean.size, min(cfg.candidate_windows), dtype=int)
    oracle_estimate = np.full(latent_mean.size, np.nan, dtype=float)
    plugin_estimate = np.full(latent_mean.size, np.nan, dtype=float)
    adaptive_estimate = np.full(latent_mean.size, np.nan, dtype=float)
    structural_estimate = np.full(latent_mean.size, np.nan, dtype=float)
    activity_estimate = np.full(latent_mean.size, np.nan, dtype=float)
    previous_adaptive_window = min(cfg.candidate_windows)
    previous_structural_proxy = 0.0
    previous_structural_window = min(cfg.candidate_windows)
    previous_activity_proxy = 0.0
    previous_activity_window = min(cfg.candidate_windows)
    for t in range(cfg.warmup, latent_mean.size):
        oracle_window[t] = oracle_window_from_latent(latent_mean, t, cfg)
        plugin = estimate_local_roughness(observations, t, cfg)
        plugin_window[t] = plugin.plugin_window
        adaptive = _adaptive_window(observations, t, cfg, previous_adaptive_window)
        adaptive_window[t] = adaptive.plugin_window
        previous_adaptive_window = adaptive.plugin_window
        structural, previous_structural_proxy = _structural_window(
            observations,
            t,
            cfg,
            previous_structural_proxy,
            previous_structural_window,
        )
        structural_window[t] = structural.plugin_window
        previous_structural_window = structural.plugin_window
        previous_activity_proxy, activity_window[t] = _activity_window(
            observations,
            t,
            cfg,
            previous_activity_proxy,
            previous_activity_window,
        )
        previous_activity_window = activity_window[t]
        oracle_estimate[t] = _moving_average(observations, t, oracle_window[t])
        plugin_estimate[t] = _moving_average(observations, t, plugin_window[t])
        adaptive_estimate[t] = _moving_average(observations, t, adaptive_window[t])
        structural_estimate[t] = _moving_average(observations, t, structural_window[t])
        activity_estimate[t] = _moving_average(observations, t, activity_window[t])
    valid = np.arange(cfg.warmup, latent_mean.size)
    static_errors: dict[int, float] = {}
    static_estimates = np.full(latent_mean.size, np.nan, dtype=float)
    best_static_window = min(cfg.candidate_windows)
    best_static_error = float("inf")
    for window in cfg.candidate_windows:
        estimates = np.full(latent_mean.size, np.nan, dtype=float)
        for t in valid:
            estimates[t] = _moving_average(observations, t, window)
        error = float(np.mean(np.abs(latent_mean[valid] - estimates[valid])))
        static_errors[window] = error
        if error < best_static_error:
            best_static_error = error
            best_static_window = window
            static_estimates = estimates
    oracle_error = np.abs(latent_mean[valid] - oracle_estimate[valid])
    plugin_error = np.abs(latent_mean[valid] - plugin_estimate[valid])
    adaptive_error = np.abs(latent_mean[valid] - adaptive_estimate[valid])
    structural_error = np.abs(latent_mean[valid] - structural_estimate[valid])
    activity_error = np.abs(latent_mean[valid] - activity_estimate[valid])
    best_static_error_trace = np.abs(latent_mean[valid] - static_estimates[valid])
    return OnlineAdaptationResult(
        config=cfg,
        time=time[valid],
        latent_mean=latent_mean[valid],
        observations=observations[valid],
        oracle_window=oracle_window[valid],
        plugin_window=plugin_window[valid],
        adaptive_window=adaptive_window[valid],
        structural_window=structural_window[valid],
        activity_window=activity_window[valid],
        plugin_estimate=plugin_estimate[valid],
        oracle_estimate=oracle_estimate[valid],
        adaptive_estimate=adaptive_estimate[valid],
        structural_estimate=structural_estimate[valid],
        activity_estimate=activity_estimate[valid],
        best_static_window=best_static_window,
        best_static_estimate=static_estimates[valid],
        oracle_error=oracle_error,
        plugin_error=plugin_error,
        adaptive_error=adaptive_error,
        structural_error=structural_error,
        activity_error=activity_error,
        best_static_error=best_static_error_trace,
        mean_oracle_error=float(np.mean(oracle_error)),
        mean_plugin_error=float(np.mean(plugin_error)),
        mean_adaptive_error=float(np.mean(adaptive_error)),
        mean_structural_error=float(np.mean(structural_error)),
        mean_activity_error=float(np.mean(activity_error)),
        mean_best_static_error=float(np.mean(best_static_error_trace)),
    )
