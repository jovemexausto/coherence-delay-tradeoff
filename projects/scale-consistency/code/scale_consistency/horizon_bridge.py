from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bridge_diagnostics import (
    dominant_periodogram_frequency,
    dominant_periodogram_power,
    cusum_squared_statistic,
    durbin_watson,
    log_variance_trend,
    quadratic_curvature_p_value,
    standardized_residuals,
    sliding_window_kl_scores,
    window_scale_test_p_values,
    variance_window_kl_scores,
)
from .estimation import feasible_wls
from .model import (
    exact_scale_profile,
    misspecified_scale_profile,
    simulate_observed_discrepancies,
)
from .theory_diagnostics import information_scale, lag_energy


def _as_lag_array(lags: int | np.ndarray) -> np.ndarray:
    if isinstance(lags, int):
        if lags < 2:
            raise ValueError("lags must be at least 2")
        return np.arange(1, lags + 1, dtype=float)
    lag_array = np.asarray(lags, dtype=float)
    if lag_array.ndim != 1 or lag_array.size < 2:
        raise ValueError("lags must be a one-dimensional array with size at least 2")
    if np.any(lag_array <= 0.0):
        raise ValueError("lags must be positive")
    return lag_array


@dataclass(frozen=True)
class LagPowerLawEstimate:
    alpha: float
    H: float
    zeta: float
    fitted: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class PercentileInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class BridgeRecoveryConfig:
    lags: int | tuple[int, ...] = 40
    n_values: tuple[int, ...] = (200, 500, 1000, 2000)
    H_values: tuple[float, ...] = (0.4, 0.6, 0.8)
    zeta_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    sigma0_values: tuple[float, ...] = (1.0,)
    C_K: float = 1.0
    C_S: float = 1.0
    a: float = 0.5
    repetitions: int = 200
    bootstrap_repetitions: int = 200
    interval_level: float = 0.95
    bootstrap_method: str = "parametric"
    bootstrap_block_length: int | None = None
    kl_window_size: int = 30
    kl_step: int = 10
    seed: int = 1234


@dataclass(frozen=True)
class BridgeRecoveryRow:
    lag_count: int
    n: int
    H: float
    zeta: float
    sigma0: float
    lag_energy: float
    information_scale: float
    true_n_star: float
    mean_H_hat: float
    bias_H: float
    rmse_H: float
    coverage_H: float
    mean_interval_width_H: float
    mean_zeta_hat: float
    bias_zeta: float
    rmse_zeta: float
    mean_n_star_hat: float
    bias_n_star: float
    rmse_n_star: float
    coverage_n_star: float
    mean_interval_width_n_star: float
    mean_residual_slope: float
    mean_durbin_watson: float
    mean_curvature_p_value: float
    mean_periodogram_peak_frequency: float
    mean_periodogram_peak_power: float
    mean_tail_kl_residual: float
    max_tail_kl_residual: float
    mean_tail_kl_standardized_residual: float
    max_tail_kl_standardized_residual: float
    mean_tail_kl_variance: float
    max_tail_kl_variance: float
    mean_tail_kl_log_observed: float
    max_tail_kl_log_observed: float
    variance_trend_slope: float
    variance_trend_p_value: float
    cusum_squared: float
    mean_levene_p_value: float
    min_levene_p_value: float
    mean_fligner_p_value: float
    min_fligner_p_value: float


@dataclass(frozen=True)
class BridgeMisspecificationConfig:
    lags: int | tuple[int, ...] = 40
    n: int = 1000
    H: float = 0.6
    zeta: float = 1.0
    sigma0: float = 1.0
    C_K: float = 1.0
    C_S: float = 1.0
    a: float = 0.5
    amplitudes: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    heteroskedastic_mode: str = "power"
    heteroskedastic_beta: float = 1.0
    heteroskedastic_jump_lag: float | None = None
    kinds: tuple[str, ...] = (
        "bump",
        "heteroskedastic",
        "mixed",
        "piecewise",
        "sinusoid",
        "slope_shift",
    )
    repetitions: int = 200
    kl_window_size: int = 30
    kl_step: int = 10
    seed: int = 4321


@dataclass(frozen=True)
class BridgeMisspecificationRow:
    lag_count: int
    kind: str
    amplitude: float
    true_n_star: float
    mean_H_hat: float
    bias_H: float
    rmse_H: float
    mean_zeta_hat: float
    bias_zeta: float
    rmse_zeta: float
    mean_n_star_hat: float
    bias_n_star: float
    rmse_n_star: float
    mean_residual_slope: float
    mean_durbin_watson: float
    mean_curvature_p_value: float
    mean_periodogram_peak_frequency: float
    mean_periodogram_peak_power: float
    mean_tail_kl_residual: float
    max_tail_kl_residual: float
    mean_tail_kl_standardized_residual: float
    max_tail_kl_standardized_residual: float
    mean_tail_kl_variance: float
    max_tail_kl_variance: float
    mean_tail_kl_log_observed: float
    max_tail_kl_log_observed: float
    variance_trend_slope: float
    variance_trend_p_value: float
    cusum_squared: float
    mean_levene_p_value: float
    min_levene_p_value: float
    mean_fligner_p_value: float
    min_fligner_p_value: float
    variance_trend_slope: float
    variance_trend_p_value: float
    cusum_squared: float
    mean_levene_p_value: float
    min_levene_p_value: float
    mean_fligner_p_value: float
    min_fligner_p_value: float


def continuous_optimal_horizon(
    C_K: float, a: float, C_S: float, zeta: float, H: float
) -> float:
    if C_K <= 0.0 or a <= 0.0 or C_S <= 0.0 or zeta <= 0.0 or H <= 0.0:
        raise ValueError("all horizon parameters must be positive")
    return float((a * C_K / (H * C_S * zeta)) ** (1.0 / (a + H)))


def fit_lag_power_law(
    observed_discrepancies: np.ndarray,
    lags: int | np.ndarray,
    *,
    sigma0: float | None,
    n: int,
) -> LagPowerLawEstimate:
    lag_array = _as_lag_array(lags)
    estimate = feasible_wls(
        np.log(np.asarray(observed_discrepancies, dtype=float)), lag_array, sigma0, n
    )
    return LagPowerLawEstimate(
        alpha=float(estimate.alpha),
        H=float(estimate.H),
        zeta=float(np.exp(estimate.alpha)),
        fitted=np.asarray(estimate.fitted, dtype=float),
        residuals=np.asarray(estimate.residuals, dtype=float),
        weights=np.asarray(estimate.weights, dtype=float),
    )


def percentile_interval(samples: np.ndarray, level: float) -> PercentileInterval:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 1 or sample_array.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    alpha = 0.5 * (1.0 - level)
    lower, upper = np.quantile(sample_array, (alpha, 1.0 - alpha))
    return PercentileInterval(lower=float(lower), upper=float(upper))


def plug_in_horizon(
    estimate: LagPowerLawEstimate,
    *,
    C_K: float,
    a: float,
    C_S: float,
) -> float:
    return continuous_optimal_horizon(C_K, a, C_S, estimate.zeta, estimate.H)


def residual_log_lag_slope(residuals: np.ndarray, lags: int | np.ndarray) -> float:
    lag_array = _as_lag_array(lags)
    slope, _intercept = np.polyfit(
        np.log(lag_array), np.asarray(residuals, dtype=float), 1
    )
    return float(slope)


def _tail_mean(scores: np.ndarray, tail_windows: int = 3) -> float:
    score_array = np.asarray(scores, dtype=float)
    if score_array.size == 0:
        return 0.0
    start = max(0, score_array.size - tail_windows)
    return float(np.mean(score_array[start:]))


def _bridge_window_diagnostics(
    observed: np.ndarray,
    residuals: np.ndarray,
    lags: np.ndarray,
    *,
    window_size: int,
    step: int,
) -> dict[str, float]:
    residual_kl_scores = sliding_window_kl_scores(
        residuals, window_size=window_size, step=step
    )
    standardized_kl_scores = sliding_window_kl_scores(
        standardized_residuals(residuals, lags), window_size=window_size, step=step
    )
    variance_kl_scores = variance_window_kl_scores(
        residuals, window_size=window_size, step=step
    )
    log_obs_kl_scores = sliding_window_kl_scores(
        np.log(observed), window_size=window_size, step=step
    )
    variance_slope, variance_p = log_variance_trend(
        residuals,
        lags,
        window_size=window_size,
        step=step,
    )
    cusum_sq = cusum_squared_statistic(residuals)
    levene_p_values, fligner_p_values = window_scale_test_p_values(
        residuals,
        window_size=window_size,
        step=step,
    )
    return {
        "mean_tail_kl_residual": _tail_mean(residual_kl_scores),
        "max_tail_kl_residual": float(np.max(residual_kl_scores)),
        "mean_tail_kl_standardized_residual": _tail_mean(standardized_kl_scores),
        "max_tail_kl_standardized_residual": float(np.max(standardized_kl_scores)),
        "mean_tail_kl_variance": _tail_mean(variance_kl_scores),
        "max_tail_kl_variance": float(np.max(variance_kl_scores)),
        "mean_tail_kl_log_observed": _tail_mean(log_obs_kl_scores),
        "max_tail_kl_log_observed": float(np.max(log_obs_kl_scores)),
        "variance_trend_slope": variance_slope,
        "variance_trend_p_value": variance_p,
        "cusum_squared": cusum_sq,
        "mean_levene_p_value": float(np.mean(levene_p_values))
        if levene_p_values.size
        else 1.0,
        "min_levene_p_value": float(np.min(levene_p_values))
        if levene_p_values.size
        else 1.0,
        "mean_fligner_p_value": float(np.mean(fligner_p_values))
        if fligner_p_values.size
        else 1.0,
        "min_fligner_p_value": float(np.min(fligner_p_values))
        if fligner_p_values.size
        else 1.0,
    }


def _wild_bootstrap_log_observations(
    estimate: LagPowerLawEstimate,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    multipliers = rng.choice(
        np.array([-1.0, 1.0], dtype=float), size=estimate.residuals.size
    )
    return estimate.fitted + estimate.residuals * multipliers


def _moving_block_bootstrap_log_observations(
    estimate: LagPowerLawEstimate,
    *,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    residuals = np.asarray(estimate.residuals, dtype=float)
    n_obs = residuals.size
    block_length = max(2, min(block_length, n_obs))
    block_starts = np.arange(0, n_obs - block_length + 1)
    sampled: list[np.ndarray] = []
    remaining = n_obs
    centered = residuals - float(np.mean(residuals))
    while remaining > 0:
        start = int(rng.choice(block_starts))
        block = centered[start : start + block_length]
        sampled.append(block)
        remaining -= block.size
    bootstrap_residuals = np.concatenate(sampled)[:n_obs]
    return estimate.fitted + bootstrap_residuals


def bootstrap_lag_power_law(
    observed_discrepancies: np.ndarray,
    lags: int | np.ndarray,
    *,
    sigma0: float,
    n: int,
    bootstrap_repetitions: int,
    interval_level: float,
    C_K: float,
    a: float,
    C_S: float,
    method: str = "parametric",
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[PercentileInterval, PercentileInterval]:
    if bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    lag_array = _as_lag_array(lags)
    bootstrap_rng = np.random.default_rng() if rng is None else rng
    estimate = fit_lag_power_law(observed_discrepancies, lag_array, sigma0=sigma0, n=n)
    H_values = np.empty(bootstrap_repetitions, dtype=float)
    n_star_values = np.empty(bootstrap_repetitions, dtype=float)
    for idx in range(bootstrap_repetitions):
        if method == "parametric":
            bootstrap_obs = simulate_observed_discrepancies(
                lag_array,
                estimate.zeta,
                estimate.H,
                sigma0,
                n,
                rng=bootstrap_rng,
            )
        elif method == "wild":
            bootstrap_obs = np.exp(
                _wild_bootstrap_log_observations(estimate, rng=bootstrap_rng)
            )
        elif method == "moving_block":
            effective_block_length = (
                max(2, int(np.sqrt(lag_array.size)))
                if block_length is None
                else block_length
            )
            bootstrap_obs = np.exp(
                _moving_block_bootstrap_log_observations(
                    estimate,
                    rng=bootstrap_rng,
                    block_length=effective_block_length,
                )
            )
        else:
            raise ValueError(f"unsupported bootstrap method: {method}")
        bootstrap_estimate = fit_lag_power_law(
            bootstrap_obs,
            lag_array,
            sigma0=sigma0,
            n=n,
        )
        H_values[idx] = bootstrap_estimate.H
        n_star_values[idx] = plug_in_horizon(
            bootstrap_estimate,
            C_K=C_K,
            a=a,
            C_S=C_S,
        )
    return (
        percentile_interval(H_values, interval_level),
        percentile_interval(n_star_values, interval_level),
    )


def run_bridge_recovery_experiment(
    config: BridgeRecoveryConfig = BridgeRecoveryConfig(),
) -> list[BridgeRecoveryRow]:
    if config.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    if not 0.0 < config.interval_level < 1.0:
        raise ValueError("interval_level must lie in (0, 1)")
    rng = np.random.default_rng(config.seed)
    lag_array = _as_lag_array(config.lags)
    rows: list[BridgeRecoveryRow] = []
    for n in config.n_values:
        for H in config.H_values:
            for zeta in config.zeta_values:
                for sigma0 in config.sigma0_values:
                    H_hats: list[float] = []
                    zeta_hats: list[float] = []
                    n_star_hats: list[float] = []
                    slopes: list[float] = []
                    H_coverages: list[float] = []
                    H_widths: list[float] = []
                    n_star_coverages: list[float] = []
                    n_star_widths: list[float] = []
                    durbin_watson_values: list[float] = []
                    curvature_p_values: list[float] = []
                    peak_frequencies: list[float] = []
                    peak_powers: list[float] = []
                    tail_kl_residuals: list[float] = []
                    max_kl_residuals: list[float] = []
                    tail_kl_standardized_residuals: list[float] = []
                    max_kl_standardized_residuals: list[float] = []
                    tail_kl_variances: list[float] = []
                    max_kl_variances: list[float] = []
                    tail_kl_logs: list[float] = []
                    max_kl_logs: list[float] = []
                    variance_trends: list[float] = []
                    variance_trend_p_values: list[float] = []
                    cusum_squared_values: list[float] = []
                    levene_p_values: list[float] = []
                    levene_min_p_values: list[float] = []
                    fligner_p_values: list[float] = []
                    fligner_min_p_values: list[float] = []
                    true_n_star = continuous_optimal_horizon(
                        config.C_K, config.a, config.C_S, zeta, H
                    )
                    for _ in range(config.repetitions):
                        obs = simulate_observed_discrepancies(
                            lag_array,
                            zeta,
                            H,
                            sigma0,
                            n,
                            rng=rng,
                        )
                        estimate = fit_lag_power_law(obs, lag_array, sigma0=sigma0, n=n)
                        H_hats.append(estimate.H)
                        zeta_hats.append(estimate.zeta)
                        n_star_hats.append(
                            plug_in_horizon(
                                estimate,
                                C_K=config.C_K,
                                a=config.a,
                                C_S=config.C_S,
                            )
                        )
                        H_interval, n_star_interval = bootstrap_lag_power_law(
                            obs,
                            lag_array,
                            sigma0=sigma0,
                            n=n,
                            bootstrap_repetitions=config.bootstrap_repetitions,
                            interval_level=config.interval_level,
                            C_K=config.C_K,
                            a=config.a,
                            C_S=config.C_S,
                            method=config.bootstrap_method,
                            block_length=config.bootstrap_block_length,
                            rng=rng,
                        )
                        slopes.append(
                            residual_log_lag_slope(estimate.residuals, lag_array)
                        )
                        durbin_watson_values.append(durbin_watson(estimate.residuals))
                        curvature_p_values.append(
                            quadratic_curvature_p_value(np.log(obs), lag_array)
                        )
                        peak_frequencies.append(
                            dominant_periodogram_frequency(estimate.residuals)
                        )
                        peak_powers.append(
                            dominant_periodogram_power(estimate.residuals)
                        )
                        window_size = min(config.kl_window_size, lag_array.size)
                        if window_size < 2:
                            raise ValueError("kl_window_size must be at least 2")
                        step = max(1, min(config.kl_step, window_size))
                        diagnostics = _bridge_window_diagnostics(
                            obs,
                            estimate.residuals,
                            lag_array,
                            window_size=window_size,
                            step=step,
                        )
                        tail_kl_residuals.append(diagnostics["mean_tail_kl_residual"])
                        max_kl_residuals.append(diagnostics["max_tail_kl_residual"])
                        tail_kl_standardized_residuals.append(
                            diagnostics["mean_tail_kl_standardized_residual"]
                        )
                        max_kl_standardized_residuals.append(
                            diagnostics["max_tail_kl_standardized_residual"]
                        )
                        tail_kl_variances.append(diagnostics["mean_tail_kl_variance"])
                        max_kl_variances.append(diagnostics["max_tail_kl_variance"])
                        tail_kl_logs.append(diagnostics["mean_tail_kl_log_observed"])
                        max_kl_logs.append(diagnostics["max_tail_kl_log_observed"])
                        variance_trends.append(diagnostics["variance_trend_slope"])
                        variance_trend_p_values.append(
                            diagnostics["variance_trend_p_value"]
                        )
                        cusum_squared_values.append(diagnostics["cusum_squared"])
                        levene_p_values.append(diagnostics["mean_levene_p_value"])
                        levene_min_p_values.append(diagnostics["min_levene_p_value"])
                        fligner_p_values.append(diagnostics["mean_fligner_p_value"])
                        fligner_min_p_values.append(diagnostics["min_fligner_p_value"])
                        H_coverages.append(
                            float(H_interval.lower <= H <= H_interval.upper)
                        )
                        H_widths.append(H_interval.upper - H_interval.lower)
                        n_star_coverages.append(
                            float(
                                n_star_interval.lower
                                <= true_n_star
                                <= n_star_interval.upper
                            )
                        )
                        n_star_widths.append(
                            n_star_interval.upper - n_star_interval.lower
                        )

                    H_hat_array = np.asarray(H_hats, dtype=float)
                    zeta_hat_array = np.asarray(zeta_hats, dtype=float)
                    n_star_hat_array = np.asarray(n_star_hats, dtype=float)
                    H_err = H_hat_array - H
                    zeta_err = zeta_hat_array - zeta
                    n_star_err = n_star_hat_array - true_n_star
                    rows.append(
                        BridgeRecoveryRow(
                            lag_count=int(lag_array.size),
                            n=n,
                            H=H,
                            zeta=zeta,
                            sigma0=sigma0,
                            lag_energy=lag_energy(lag_array, H),
                            information_scale=information_scale(n, lag_array, H),
                            true_n_star=true_n_star,
                            mean_H_hat=float(np.mean(H_hat_array)),
                            bias_H=float(np.mean(H_err)),
                            rmse_H=float(np.sqrt(np.mean(H_err**2))),
                            coverage_H=float(np.mean(H_coverages)),
                            mean_interval_width_H=float(np.mean(H_widths)),
                            mean_zeta_hat=float(np.mean(zeta_hat_array)),
                            bias_zeta=float(np.mean(zeta_err)),
                            rmse_zeta=float(np.sqrt(np.mean(zeta_err**2))),
                            mean_n_star_hat=float(np.mean(n_star_hat_array)),
                            bias_n_star=float(np.mean(n_star_err)),
                            rmse_n_star=float(np.sqrt(np.mean(n_star_err**2))),
                            coverage_n_star=float(np.mean(n_star_coverages)),
                            mean_interval_width_n_star=float(np.mean(n_star_widths)),
                            mean_residual_slope=float(np.mean(slopes)),
                            mean_durbin_watson=float(np.mean(durbin_watson_values)),
                            mean_curvature_p_value=float(np.mean(curvature_p_values)),
                            mean_periodogram_peak_frequency=float(
                                np.mean(peak_frequencies)
                            ),
                            mean_periodogram_peak_power=float(np.mean(peak_powers)),
                            mean_tail_kl_residual=float(np.mean(tail_kl_residuals)),
                            max_tail_kl_residual=float(np.mean(max_kl_residuals)),
                            mean_tail_kl_standardized_residual=float(
                                np.mean(tail_kl_standardized_residuals)
                            ),
                            max_tail_kl_standardized_residual=float(
                                np.mean(max_kl_standardized_residuals)
                            ),
                            mean_tail_kl_variance=float(np.mean(tail_kl_variances)),
                            max_tail_kl_variance=float(np.mean(max_kl_variances)),
                            mean_tail_kl_log_observed=float(np.mean(tail_kl_logs)),
                            max_tail_kl_log_observed=float(np.mean(max_kl_logs)),
                            variance_trend_slope=float(np.mean(variance_trends)),
                            variance_trend_p_value=float(
                                np.mean(variance_trend_p_values)
                            ),
                            cusum_squared=float(np.mean(cusum_squared_values)),
                            mean_levene_p_value=float(np.mean(levene_p_values)),
                            min_levene_p_value=float(np.mean(levene_min_p_values)),
                            mean_fligner_p_value=float(np.mean(fligner_p_values)),
                            min_fligner_p_value=float(np.mean(fligner_min_p_values)),
                        )
                    )
    return rows


def run_bridge_misspecification_experiment(
    config: BridgeMisspecificationConfig = BridgeMisspecificationConfig(),
) -> list[BridgeMisspecificationRow]:
    rng = np.random.default_rng(config.seed)
    lag_array = _as_lag_array(config.lags)
    rows: list[BridgeMisspecificationRow] = []
    true_n_star = continuous_optimal_horizon(
        config.C_K, config.a, config.C_S, config.zeta, config.H
    )
    for kind in config.kinds:
        for amplitude in config.amplitudes:
            H_hats: list[float] = []
            zeta_hats: list[float] = []
            n_star_hats: list[float] = []
            slopes: list[float] = []
            durbin_watson_values: list[float] = []
            curvature_p_values: list[float] = []
            peak_frequencies: list[float] = []
            peak_powers: list[float] = []
            tail_kl_residuals: list[float] = []
            max_kl_residuals: list[float] = []
            tail_kl_standardized_residuals: list[float] = []
            max_kl_standardized_residuals: list[float] = []
            tail_kl_variances: list[float] = []
            max_kl_variances: list[float] = []
            tail_kl_logs: list[float] = []
            max_kl_logs: list[float] = []
            variance_trends: list[float] = []
            variance_trend_p_values: list[float] = []
            cusum_squared_values: list[float] = []
            levene_p_values: list[float] = []
            levene_min_p_values: list[float] = []
            fligner_p_values: list[float] = []
            fligner_min_p_values: list[float] = []
            profile = (
                exact_scale_profile(lag_array, config.zeta, config.H)
                if kind == "heteroskedastic"
                else misspecified_scale_profile(
                    lag_array,
                    config.zeta,
                    config.H,
                    amplitude,
                    kind=kind,
                )
            )
            for _ in range(config.repetitions):
                if kind == "heteroskedastic":
                    noise_kind = f"heteroskedastic_{config.heteroskedastic_mode}"
                    noise_kappa = amplitude
                else:
                    noise_kind = "gaussian"
                    noise_kappa = 0.0
                obs = simulate_observed_discrepancies(
                    lag_array,
                    config.zeta,
                    config.H,
                    config.sigma0,
                    config.n,
                    kappa=noise_kappa,
                    rng=rng,
                    profile=profile,
                    noise=noise_kind,
                    heteroskedastic_alpha=amplitude,
                    heteroskedastic_beta=config.heteroskedastic_beta,
                    heteroskedastic_jump_lag=config.heteroskedastic_jump_lag,
                )
                estimate = fit_lag_power_law(
                    obs, lag_array, sigma0=config.sigma0, n=config.n
                )
                H_hats.append(estimate.H)
                zeta_hats.append(estimate.zeta)
                n_star_hats.append(
                    plug_in_horizon(
                        estimate,
                        C_K=config.C_K,
                        a=config.a,
                        C_S=config.C_S,
                    )
                )
                slopes.append(residual_log_lag_slope(estimate.residuals, lag_array))
                durbin_watson_values.append(durbin_watson(estimate.residuals))
                curvature_p_values.append(
                    quadratic_curvature_p_value(np.log(obs), lag_array)
                )
                peak_frequencies.append(
                    dominant_periodogram_frequency(estimate.residuals)
                )
                peak_powers.append(dominant_periodogram_power(estimate.residuals))
                window_size = min(config.kl_window_size, lag_array.size)
                if window_size < 2:
                    raise ValueError("kl_window_size must be at least 2")
                step = max(1, min(config.kl_step, window_size))
                diagnostics = _bridge_window_diagnostics(
                    obs,
                    estimate.residuals,
                    lag_array,
                    window_size=window_size,
                    step=step,
                )
                tail_kl_residuals.append(diagnostics["mean_tail_kl_residual"])
                max_kl_residuals.append(diagnostics["max_tail_kl_residual"])
                tail_kl_standardized_residuals.append(
                    diagnostics["mean_tail_kl_standardized_residual"]
                )
                max_kl_standardized_residuals.append(
                    diagnostics["max_tail_kl_standardized_residual"]
                )
                tail_kl_variances.append(diagnostics["mean_tail_kl_variance"])
                max_kl_variances.append(diagnostics["max_tail_kl_variance"])
                tail_kl_logs.append(diagnostics["mean_tail_kl_log_observed"])
                max_kl_logs.append(diagnostics["max_tail_kl_log_observed"])
                variance_trends.append(diagnostics["variance_trend_slope"])
                variance_trend_p_values.append(diagnostics["variance_trend_p_value"])
                cusum_squared_values.append(diagnostics["cusum_squared"])
                levene_p_values.append(diagnostics["mean_levene_p_value"])
                levene_min_p_values.append(diagnostics["min_levene_p_value"])
                fligner_p_values.append(diagnostics["mean_fligner_p_value"])
                fligner_min_p_values.append(diagnostics["min_fligner_p_value"])

            H_hat_array = np.asarray(H_hats, dtype=float)
            zeta_hat_array = np.asarray(zeta_hats, dtype=float)
            n_star_hat_array = np.asarray(n_star_hats, dtype=float)
            H_err = H_hat_array - config.H
            zeta_err = zeta_hat_array - config.zeta
            n_star_err = n_star_hat_array - true_n_star
            rows.append(
                BridgeMisspecificationRow(
                    lag_count=int(lag_array.size),
                    kind=kind,
                    amplitude=amplitude,
                    true_n_star=true_n_star,
                    mean_H_hat=float(np.mean(H_hat_array)),
                    bias_H=float(np.mean(H_err)),
                    rmse_H=float(np.sqrt(np.mean(H_err**2))),
                    mean_zeta_hat=float(np.mean(zeta_hat_array)),
                    bias_zeta=float(np.mean(zeta_err)),
                    rmse_zeta=float(np.sqrt(np.mean(zeta_err**2))),
                    mean_n_star_hat=float(np.mean(n_star_hat_array)),
                    bias_n_star=float(np.mean(n_star_err)),
                    rmse_n_star=float(np.sqrt(np.mean(n_star_err**2))),
                    mean_residual_slope=float(np.mean(slopes)),
                    mean_durbin_watson=float(np.mean(durbin_watson_values)),
                    mean_curvature_p_value=float(np.mean(curvature_p_values)),
                    mean_periodogram_peak_frequency=float(np.mean(peak_frequencies)),
                    mean_periodogram_peak_power=float(np.mean(peak_powers)),
                    mean_tail_kl_residual=float(np.mean(tail_kl_residuals)),
                    max_tail_kl_residual=float(np.mean(max_kl_residuals)),
                    mean_tail_kl_standardized_residual=float(
                        np.mean(tail_kl_standardized_residuals)
                    ),
                    max_tail_kl_standardized_residual=float(
                        np.mean(max_kl_standardized_residuals)
                    ),
                    mean_tail_kl_variance=float(np.mean(tail_kl_variances)),
                    max_tail_kl_variance=float(np.mean(max_kl_variances)),
                    mean_tail_kl_log_observed=float(np.mean(tail_kl_logs)),
                    max_tail_kl_log_observed=float(np.mean(max_kl_logs)),
                    variance_trend_slope=float(np.mean(variance_trends)),
                    variance_trend_p_value=float(np.mean(variance_trend_p_values)),
                    cusum_squared=float(np.mean(cusum_squared_values)),
                    mean_levene_p_value=float(np.mean(levene_p_values)),
                    min_levene_p_value=float(np.mean(levene_min_p_values)),
                    mean_fligner_p_value=float(np.mean(fligner_p_values)),
                    min_fligner_p_value=float(np.mean(fligner_min_p_values)),
                )
            )
    return rows
