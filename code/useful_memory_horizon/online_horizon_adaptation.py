from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import build_manifest_row, export_rows_csv, stable_run_id


def _default_candidate_windows() -> tuple[int, ...]:
    values = np.unique(np.round(np.geomspace(8, 256, 18)).astype(int)).tolist()
    return tuple(int(value) for value in values)


@dataclass(frozen=True, slots=True)
class OnlineAdaptationConfig:
    profile_name: str = "default"
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
    phase_index: np.ndarray
    holder_truth: np.ndarray
    roughness_truth: np.ndarray
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
    plugin_holder_hat: np.ndarray
    plugin_roughness_hat: np.ndarray
    adaptive_holder_hat: np.ndarray
    adaptive_roughness_hat: np.ndarray
    structural_holder_hat: np.ndarray
    structural_roughness_hat: np.ndarray
    activity_proxy: np.ndarray
    structural_band_min: np.ndarray
    structural_band_max: np.ndarray
    adaptive_validation_score: np.ndarray
    structural_validation_score: np.ndarray
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


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    chosen_window: int
    holder_hat: float
    roughness_hat: float
    activity_proxy: float
    band_min: int
    band_max: int
    validation_score: float


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _phase_index_array(config: OnlineAdaptationConfig) -> np.ndarray:
    phase_index = np.empty(sum(config.phase_lengths), dtype=int)
    start = 0
    for phase, length in enumerate(config.phase_lengths):
        phase_index[start : start + length] = phase
        start += length
    return phase_index


def make_profile_config(
    profile_name: str,
    *,
    seed: int,
    observation_scale: float = 1.0,
    aggregated_history: int = 160,
    validation_tail: int = 160,
    max_window_index_jump: int | None = None,
) -> OnlineAdaptationConfig:
    profiles: dict[str, dict[str, tuple[float, ...] | tuple[int, ...]]] = {
        "default": {
            "phase_lengths": (300, 300, 300, 300),
            "holder_exponents": (1.0, 0.75, 0.5, 1.0),
            "roughness_scales": (0.001, 0.08, 0.001, 0.08),
            "phase_signs": (1.0, -1.0, 1.0, -1.0),
        },
        "smooth": {
            "phase_lengths": (300, 300, 300, 300),
            "holder_exponents": (1.0, 1.0, 0.9, 1.0),
            "roughness_scales": (0.001, 0.01, 0.002, 0.01),
            "phase_signs": (1.0, -1.0, 1.0, -1.0),
        },
        "rough": {
            "phase_lengths": (300, 300, 300, 300),
            "holder_exponents": (0.55, 0.45, 0.4, 0.5),
            "roughness_scales": (0.03, 0.08, 0.04, 0.08),
            "phase_signs": (1.0, -1.0, 1.0, -1.0),
        },
        "alternating": {
            "phase_lengths": (240, 240, 240, 240, 240),
            "holder_exponents": (1.0, 0.5, 1.0, 0.5, 1.0),
            "roughness_scales": (0.002, 0.06, 0.002, 0.06, 0.002),
            "phase_signs": (1.0, -1.0, 1.0, -1.0, 1.0),
        },
        "ramp_up": {
            "phase_lengths": (300, 300, 300, 300),
            "holder_exponents": (1.0, 0.85, 0.7, 0.55),
            "roughness_scales": (0.001, 0.01, 0.03, 0.08),
            "phase_signs": (1.0, 1.0, 1.0, 1.0),
        },
    }
    if profile_name not in profiles:
        raise ValueError(f"unknown profile_name: {profile_name}")
    profile = profiles[profile_name]
    return OnlineAdaptationConfig(
        profile_name=profile_name,
        phase_lengths=profile["phase_lengths"],  # type: ignore[arg-type]
        holder_exponents=profile["holder_exponents"],  # type: ignore[arg-type]
        roughness_scales=profile["roughness_scales"],  # type: ignore[arg-type]
        phase_signs=profile["phase_signs"],  # type: ignore[arg-type]
        observation_scale=observation_scale,
        aggregated_history=aggregated_history,
        validation_tail=validation_tail,
        max_window_index_jump=max_window_index_jump,
        seed=seed,
    )


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
) -> ControllerDecision:
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
    best_score = float(
        scores.get(
            best_window,
            _validation_score(values, end_index, best_window, config.validation_tail),
        )
    )
    return ControllerDecision(
        chosen_window=best_window,
        holder_hat=roughness.holder_exponent,
        roughness_hat=roughness.roughness_scale,
        activity_proxy=activity_proxy,
        band_min=int(min(feasible_windows)),
        band_max=int(max(feasible_windows)),
        validation_score=best_score,
    )


def _adaptive_window(
    values: np.ndarray,
    end_index: int,
    config: OnlineAdaptationConfig,
    previous_window: int,
) -> ControllerDecision:
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
    return ControllerDecision(
        chosen_window=best_window,
        holder_hat=estimate.holder_exponent,
        roughness_hat=estimate.roughness_scale,
        activity_proxy=float("nan"),
        band_min=int(best_window),
        band_max=int(best_window),
        validation_score=float(scores[best_window]),
    )


def run_online_horizon_adaptation_experiment(
    config: OnlineAdaptationConfig | None = None,
) -> OnlineAdaptationResult:
    cfg = config or OnlineAdaptationConfig()
    latent_mean, holder_truth, roughness_truth = phase_profile(cfg)
    phase_index = _phase_index_array(cfg)
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
    plugin_holder_hat = np.full(latent_mean.size, np.nan, dtype=float)
    plugin_roughness_hat = np.full(latent_mean.size, np.nan, dtype=float)
    adaptive_holder_hat = np.full(latent_mean.size, np.nan, dtype=float)
    adaptive_roughness_hat = np.full(latent_mean.size, np.nan, dtype=float)
    structural_holder_hat = np.full(latent_mean.size, np.nan, dtype=float)
    structural_roughness_hat = np.full(latent_mean.size, np.nan, dtype=float)
    activity_proxy_trace = np.full(latent_mean.size, np.nan, dtype=float)
    structural_band_min = np.full(latent_mean.size, np.nan, dtype=float)
    structural_band_max = np.full(latent_mean.size, np.nan, dtype=float)
    adaptive_validation_score = np.full(latent_mean.size, np.nan, dtype=float)
    structural_validation_score = np.full(latent_mean.size, np.nan, dtype=float)
    previous_adaptive_window = min(cfg.candidate_windows)
    previous_structural_proxy = 0.0
    previous_structural_window = min(cfg.candidate_windows)
    previous_activity_proxy = 0.0
    previous_activity_window = min(cfg.candidate_windows)
    for t in range(cfg.warmup, latent_mean.size):
        oracle_window[t] = oracle_window_from_latent(latent_mean, t, cfg)
        plugin = estimate_local_roughness(observations, t, cfg)
        plugin_window[t] = plugin.plugin_window
        plugin_holder_hat[t] = plugin.holder_exponent
        plugin_roughness_hat[t] = plugin.roughness_scale
        adaptive = _adaptive_window(observations, t, cfg, previous_adaptive_window)
        adaptive_window[t] = adaptive.chosen_window
        adaptive_holder_hat[t] = adaptive.holder_hat
        adaptive_roughness_hat[t] = adaptive.roughness_hat
        adaptive_validation_score[t] = adaptive.validation_score
        previous_adaptive_window = adaptive.chosen_window
        structural = _structural_window(
            observations,
            t,
            cfg,
            previous_structural_proxy,
            previous_structural_window,
        )
        structural_window[t] = structural.chosen_window
        structural_holder_hat[t] = structural.holder_hat
        structural_roughness_hat[t] = structural.roughness_hat
        structural_band_min[t] = structural.band_min
        structural_band_max[t] = structural.band_max
        structural_validation_score[t] = structural.validation_score
        activity_proxy_trace[t] = structural.activity_proxy
        previous_structural_proxy = structural.activity_proxy
        previous_structural_window = structural.chosen_window
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
        phase_index=phase_index[valid],
        holder_truth=holder_truth[valid],
        roughness_truth=roughness_truth[valid],
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
        plugin_holder_hat=plugin_holder_hat[valid],
        plugin_roughness_hat=plugin_roughness_hat[valid],
        adaptive_holder_hat=adaptive_holder_hat[valid],
        adaptive_roughness_hat=adaptive_roughness_hat[valid],
        structural_holder_hat=structural_holder_hat[valid],
        structural_roughness_hat=structural_roughness_hat[valid],
        activity_proxy=activity_proxy_trace[valid],
        structural_band_min=structural_band_min[valid],
        structural_band_max=structural_band_max[valid],
        adaptive_validation_score=adaptive_validation_score[valid],
        structural_validation_score=structural_validation_score[valid],
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


def build_online_summary_rows(
    results: list[OnlineAdaptationResult],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        config = asdict(result.config)
        run_id = stable_run_id(config)
        rows.append(
            {
                "run_id": run_id,
                "profile_name": result.config.profile_name,
                "seed": result.config.seed,
                "observation_scale": result.config.observation_scale,
                "aggregated_history": result.config.aggregated_history,
                "validation_tail": result.config.validation_tail,
                "max_window_index_jump": ""
                if result.config.max_window_index_jump is None
                else result.config.max_window_index_jump,
                "best_static_window": result.best_static_window,
                "mean_oracle_error": round(result.mean_oracle_error, 6),
                "mean_plugin_error": round(result.mean_plugin_error, 6),
                "mean_activity_error": round(result.mean_activity_error, 6),
                "mean_structural_error": round(result.mean_structural_error, 6),
                "mean_adaptive_error": round(result.mean_adaptive_error, 6),
                "mean_best_static_error": round(result.mean_best_static_error, 6),
                "plugin_to_oracle_ratio": round(
                    result.mean_plugin_error / result.mean_oracle_error, 6
                ),
                "activity_to_oracle_ratio": round(
                    result.mean_activity_error / result.mean_oracle_error, 6
                ),
                "structural_to_oracle_ratio": round(
                    result.mean_structural_error / result.mean_oracle_error, 6
                ),
                "adaptive_to_oracle_ratio": round(
                    result.mean_adaptive_error / result.mean_oracle_error, 6
                ),
                "structural_beats_plugin": int(
                    result.mean_structural_error < result.mean_plugin_error
                ),
                "structural_beats_activity": int(
                    result.mean_structural_error < result.mean_activity_error
                ),
                "adaptive_beats_structural": int(
                    result.mean_adaptive_error < result.mean_structural_error
                ),
            }
        )
    return rows


def build_online_phase_rows(
    results: list[OnlineAdaptationResult],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        run_id = stable_run_id(asdict(result.config))
        for phase in np.unique(result.phase_index):
            mask = result.phase_index == phase
            rows.append(
                {
                    "run_id": run_id,
                    "profile_name": result.config.profile_name,
                    "phase_index": int(phase),
                    "phase_length": int(np.sum(mask)),
                    "holder_truth": round(float(np.mean(result.holder_truth[mask])), 6),
                    "roughness_truth": round(
                        float(np.mean(result.roughness_truth[mask])), 6
                    ),
                    "mean_oracle_window": round(
                        float(np.mean(result.oracle_window[mask])), 3
                    ),
                    "mean_plugin_window": round(
                        float(np.mean(result.plugin_window[mask])), 3
                    ),
                    "mean_activity_window": round(
                        float(np.mean(result.activity_window[mask])), 3
                    ),
                    "mean_structural_window": round(
                        float(np.mean(result.structural_window[mask])), 3
                    ),
                    "mean_adaptive_window": round(
                        float(np.mean(result.adaptive_window[mask])), 3
                    ),
                    "mean_oracle_error": round(
                        float(np.mean(result.oracle_error[mask])), 6
                    ),
                    "mean_plugin_error": round(
                        float(np.mean(result.plugin_error[mask])), 6
                    ),
                    "mean_activity_error": round(
                        float(np.mean(result.activity_error[mask])), 6
                    ),
                    "mean_structural_error": round(
                        float(np.mean(result.structural_error[mask])), 6
                    ),
                    "mean_adaptive_error": round(
                        float(np.mean(result.adaptive_error[mask])), 6
                    ),
                }
            )
    return rows


def build_online_timeline_rows(
    result: OnlineAdaptationResult,
    *,
    stride: int = 1,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    run_id = stable_run_id(asdict(result.config))
    for idx in range(0, result.time.size, max(stride, 1)):
        rows.append(
            {
                "run_id": run_id,
                "time": int(result.time[idx]),
                "phase_index": int(result.phase_index[idx]),
                "latent_mean": round(float(result.latent_mean[idx]), 6),
                "observation": round(float(result.observations[idx]), 6),
                "holder_truth": round(float(result.holder_truth[idx]), 6),
                "roughness_truth": round(float(result.roughness_truth[idx]), 6),
                "oracle_window": int(result.oracle_window[idx]),
                "plugin_window": int(result.plugin_window[idx]),
                "activity_window": int(result.activity_window[idx]),
                "structural_window": int(result.structural_window[idx]),
                "adaptive_window": int(result.adaptive_window[idx]),
                "oracle_error": round(float(result.oracle_error[idx]), 6),
                "plugin_error": round(float(result.plugin_error[idx]), 6),
                "activity_error": round(float(result.activity_error[idx]), 6),
                "structural_error": round(float(result.structural_error[idx]), 6),
                "adaptive_error": round(float(result.adaptive_error[idx]), 6),
                "plugin_holder_hat": round(float(result.plugin_holder_hat[idx]), 6),
                "plugin_roughness_hat": round(
                    float(result.plugin_roughness_hat[idx]), 6
                ),
                "structural_holder_hat": round(
                    float(result.structural_holder_hat[idx]), 6
                ),
                "structural_roughness_hat": round(
                    float(result.structural_roughness_hat[idx]), 6
                ),
                "adaptive_holder_hat": round(float(result.adaptive_holder_hat[idx]), 6),
                "adaptive_roughness_hat": round(
                    float(result.adaptive_roughness_hat[idx]), 6
                ),
                "activity_proxy": round(float(result.activity_proxy[idx]), 6),
                "structural_band_min": ""
                if np.isnan(result.structural_band_min[idx])
                else int(result.structural_band_min[idx]),
                "structural_band_max": ""
                if np.isnan(result.structural_band_max[idx])
                else int(result.structural_band_max[idx]),
                "structural_validation_score": round(
                    float(result.structural_validation_score[idx]), 6
                ),
                "adaptive_validation_score": round(
                    float(result.adaptive_validation_score[idx]), 6
                ),
            }
        )
    return rows


def build_online_adaptation_configs(
    *,
    profile_names: tuple[str, ...],
    seeds: tuple[int, ...],
    observation_scales: tuple[float, ...],
    aggregated_histories: tuple[int, ...],
    validation_tails: tuple[int, ...],
    max_window_index_jump_values: tuple[int | None, ...],
) -> list[OnlineAdaptationConfig]:
    configs: list[OnlineAdaptationConfig] = []
    for profile_name in profile_names:
        for seed in seeds:
            for observation_scale in observation_scales:
                for aggregated_history in aggregated_histories:
                    for validation_tail in validation_tails:
                        for max_jump in max_window_index_jump_values:
                            configs.append(
                                make_profile_config(
                                    profile_name,
                                    seed=seed,
                                    observation_scale=observation_scale,
                                    aggregated_history=aggregated_history,
                                    validation_tail=validation_tail,
                                    max_window_index_jump=max_jump,
                                )
                            )
    return configs


def run_online_adaptation_sweep(
    configs: list[OnlineAdaptationConfig],
) -> list[OnlineAdaptationResult]:
    return [run_online_horizon_adaptation_experiment(config) for config in configs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate online adaptation artifacts."
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/online_adaptation")
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="default",
        help="Comma-separated profile names.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--observation-scales",
        type=str,
        default="1.0",
        help="Comma-separated observation noise scales.",
    )
    parser.add_argument(
        "--aggregated-histories",
        type=str,
        default="160",
        help="Comma-separated aggregated-history settings.",
    )
    parser.add_argument(
        "--validation-tails",
        type=str,
        default="160",
        help="Comma-separated validation-tail settings.",
    )
    parser.add_argument(
        "--max-window-index-jumps",
        type=str,
        default="",
        help="Comma-separated index-jump caps; empty means no cap.",
    )
    parser.add_argument(
        "--timeline-stride",
        type=int,
        default=8,
        help="Stride for timeline export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    max_jump_values: tuple[int | None, ...]
    if args.max_window_index_jumps:
        max_jump_values = tuple(_parse_csv_ints(args.max_window_index_jumps))
    else:
        max_jump_values = (None,)
    configs = build_online_adaptation_configs(
        profile_names=_parse_csv_strings(args.profiles),
        seeds=_parse_csv_ints(args.seeds),
        observation_scales=_parse_csv_floats(args.observation_scales),
        aggregated_histories=_parse_csv_ints(args.aggregated_histories),
        validation_tails=_parse_csv_ints(args.validation_tails),
        max_window_index_jump_values=max_jump_values,
    )
    results = run_online_adaptation_sweep(configs)
    export_rows_csv(build_online_summary_rows(results), args.csv_dir / "summary.csv")
    export_rows_csv(
        build_online_phase_rows(results), args.csv_dir / "phase_summary.csv"
    )
    timeline_rows: list[dict[str, float | int | str]] = []
    for result in results:
        timeline_rows.extend(
            build_online_timeline_rows(result, stride=args.timeline_stride)
        )
    export_rows_csv(timeline_rows, args.csv_dir / "timeline.csv")
    manifest = build_manifest_row(
        "online_adaptation",
        {
            "profiles": _parse_csv_strings(args.profiles),
            "seeds": _parse_csv_ints(args.seeds),
            "observation_scales": _parse_csv_floats(args.observation_scales),
            "aggregated_histories": _parse_csv_ints(args.aggregated_histories),
            "validation_tails": _parse_csv_ints(args.validation_tails),
            "max_window_index_jumps": max_jump_values,
            "timeline_stride": args.timeline_stride,
        },
        run_id=stable_run_id(
            {
                "profiles": args.profiles,
                "seeds": args.seeds,
                "observation_scales": args.observation_scales,
                "aggregated_histories": args.aggregated_histories,
                "validation_tails": args.validation_tails,
                "max_window_index_jumps": args.max_window_index_jumps,
                "timeline_stride": args.timeline_stride,
            }
        ),
        notes="Online adaptation sweep with summary, phase, and timeline exports.",
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")
