from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import chi2, f

from .model import simulate_observed_discrepancies


@dataclass(frozen=True)
class RegressionEstimate:
    alpha: float
    H: float
    fitted: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray
    sigma0_hat: float | None = None


@dataclass(frozen=True)
class ScaleConsistencyTestResult:
    statistic: float
    critical_value: float
    reject: bool
    degrees_of_freedom: int
    calibration: str
    estimate: RegressionEstimate


@dataclass(frozen=True)
class SplitScaleConsistencyTestResult:
    statistic: float
    critical_value: float
    reject: bool
    numerator_degrees_of_freedom: int
    denominator_degrees_of_freedom: int
    calibration: str
    scale_estimate: RegressionEstimate
    test_estimate: RegressionEstimate


def design_matrix(lags: np.ndarray) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    return np.column_stack([np.ones(lag_array.size, dtype=float), np.log(lag_array)])


def weighted_least_squares(
    y: np.ndarray,
    lags: np.ndarray,
    weights: np.ndarray,
) -> RegressionEstimate:
    y_array = np.asarray(y, dtype=float)
    lag_array = np.asarray(lags, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if y_array.shape != lag_array.shape or y_array.shape != weight_array.shape:
        raise ValueError("y, lags, and weights must have identical shapes")
    if np.any(weight_array <= 0.0):
        raise ValueError("weights must be positive")
    X = design_matrix(lag_array)
    sqrt_w = np.sqrt(weight_array)
    Xw = X * sqrt_w[:, None]
    yw = y_array * sqrt_w
    beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    fitted = X @ beta
    residuals = y_array - fitted
    return RegressionEstimate(
        alpha=float(beta[0]),
        H=float(beta[1]),
        fitted=fitted,
        residuals=residuals,
        weights=weight_array,
    )


def pilot_ols(log_observations: np.ndarray, lags: np.ndarray) -> RegressionEstimate:
    ones = np.ones_like(np.asarray(log_observations, dtype=float))
    return weighted_least_squares(log_observations, lags, ones)


def estimate_sigma0_squared_from_pilot(
    pilot: RegressionEstimate,
    n: int,
) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    fitted_scale = np.exp(np.asarray(pilot.fitted, dtype=float))
    residuals = np.asarray(pilot.residuals, dtype=float)
    df = max(int(residuals.size) - 2, 1)
    sigma0_sq_hat = (
        float(n) * float(np.sum((fitted_scale * residuals) ** 2)) / float(df)
    )
    if not np.isfinite(sigma0_sq_hat) or sigma0_sq_hat <= 0.0:
        raise ValueError("estimated sigma0^2 must be positive and finite")
    return sigma0_sq_hat


def oracle_precision_weights(
    lags: np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    D = zeta * lag_array**H
    return float(n) * D**2 / sigma0**2


def feasible_precision_weights(
    lags: np.ndarray,
    alpha: float,
    H: float,
    sigma0: float,
    n: int,
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    D_hat = np.exp(alpha) * lag_array**H
    return float(n) * D_hat**2 / sigma0**2


def oracle_wls(
    log_observations: np.ndarray,
    lags: np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
) -> RegressionEstimate:
    weights = oracle_precision_weights(lags, zeta, H, sigma0, n)
    return weighted_least_squares(log_observations, lags, weights)


def feasible_wls(
    log_observations: np.ndarray,
    lags: np.ndarray,
    sigma0: float | None,
    n: int,
) -> RegressionEstimate:
    pilot = pilot_ols(log_observations, lags)
    sigma0_hat = None
    sigma0_eff = float(sigma0) if sigma0 is not None else None
    if sigma0_eff is None:
        sigma0_sq_hat = estimate_sigma0_squared_from_pilot(pilot, n)
        sigma0_eff = float(np.sqrt(sigma0_sq_hat))
        sigma0_hat = sigma0_eff
    weights = feasible_precision_weights(lags, pilot.alpha, pilot.H, sigma0_eff, n)
    estimate = weighted_least_squares(log_observations, lags, weights)
    return RegressionEstimate(
        alpha=estimate.alpha,
        H=estimate.H,
        fitted=estimate.fitted,
        residuals=estimate.residuals,
        weights=estimate.weights,
        sigma0_hat=sigma0_hat,
    )


def residual_statistic(residuals: np.ndarray, weights: np.ndarray) -> float:
    residual_array = np.asarray(residuals, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return float(np.sum(weight_array * residual_array**2))


def _fit_test_statistic(
    observed_discrepancies: np.ndarray,
    lags: np.ndarray,
    sigma0: float | None,
    n: int,
) -> tuple[RegressionEstimate, float]:
    observed = np.asarray(observed_discrepancies, dtype=float)
    if np.any(observed <= 0.0):
        raise ValueError("observed discrepancies must be positive")
    estimate = feasible_wls(np.log(observed), np.asarray(lags, dtype=float), sigma0, n)
    statistic = residual_statistic(estimate.residuals, estimate.weights)
    return estimate, statistic


def run_scale_consistency_test(
    observed_discrepancies: np.ndarray,
    lags: np.ndarray,
    sigma0: float | None,
    n: int,
    *,
    alpha_level: float = 0.05,
    calibration: Literal["chi2", "bootstrap"] | None = None,
    bootstrap_repetitions: int = 0,
    rng: np.random.Generator | None = None,
) -> ScaleConsistencyTestResult:
    if not 0.0 < alpha_level < 1.0:
        raise ValueError("alpha_level must lie in (0, 1)")
    observed = np.asarray(observed_discrepancies, dtype=float)
    if calibration is None:
        calibration = (
            "bootstrap" if sigma0 is None and bootstrap_repetitions > 0 else "chi2"
        )
    estimate, statistic = _fit_test_statistic(observed, lags, sigma0, n)
    df = observed.size - 2 if sigma0 is not None else observed.size - 3
    if calibration == "chi2":
        critical_value = float(chi2.ppf(1.0 - alpha_level, df=df))
    elif calibration == "bootstrap":
        if bootstrap_repetitions <= 0:
            raise ValueError("bootstrap_repetitions must be positive")
        bootstrap_rng = np.random.default_rng() if rng is None else rng
        zeta_hat = float(np.exp(estimate.alpha))
        sigma0_boot = float(estimate.sigma0_hat if sigma0 is None else sigma0)
        bootstrap_statistics = np.empty(bootstrap_repetitions, dtype=float)
        lag_array = np.asarray(lags, dtype=float)
        for idx in range(bootstrap_repetitions):
            boot_obs = simulate_observed_discrepancies(
                lag_array,
                zeta_hat,
                estimate.H,
                sigma0_boot,
                n,
                rng=bootstrap_rng,
            )
            _, bootstrap_statistics[idx] = _fit_test_statistic(
                boot_obs, lag_array, sigma0, n
            )
        critical_value = float(np.quantile(bootstrap_statistics, 1.0 - alpha_level))
    else:
        raise ValueError(f"unsupported calibration: {calibration}")
    return ScaleConsistencyTestResult(
        statistic=statistic,
        critical_value=critical_value,
        reject=statistic > critical_value,
        degrees_of_freedom=df,
        calibration=calibration,
        estimate=estimate,
    )


def run_split_scale_consistency_test(
    scale_observed_discrepancies: np.ndarray,
    test_observed_discrepancies: np.ndarray,
    lags: np.ndarray,
    n_scale: int,
    n_test: int,
    *,
    alpha_level: float = 0.05,
    calibration: Literal["f"] = "f",
) -> SplitScaleConsistencyTestResult:
    if not 0.0 < alpha_level < 1.0:
        raise ValueError("alpha_level must lie in (0, 1)")
    if calibration != "f":
        raise ValueError(f"unsupported calibration: {calibration}")
    lag_array = np.asarray(lags, dtype=float)
    scale_estimate = feasible_wls(
        np.log(np.asarray(scale_observed_discrepancies, dtype=float)),
        lag_array,
        None,
        n_scale,
    )
    if scale_estimate.sigma0_hat is None:
        raise ValueError("split test requires a plug-in scale estimate")
    test_estimate = feasible_wls(
        np.log(np.asarray(test_observed_discrepancies, dtype=float)),
        lag_array,
        scale_estimate.sigma0_hat,
        n_test,
    )
    statistic = residual_statistic(test_estimate.residuals, test_estimate.weights)
    numerator_df = lag_array.size - 2
    denominator_df = lag_array.size - 2
    critical_value = float(
        numerator_df * f.ppf(1.0 - alpha_level, numerator_df, denominator_df)
    )
    return SplitScaleConsistencyTestResult(
        statistic=statistic,
        critical_value=critical_value,
        reject=statistic > critical_value,
        numerator_degrees_of_freedom=numerator_df,
        denominator_degrees_of_freedom=denominator_df,
        calibration=calibration,
        scale_estimate=scale_estimate,
        test_estimate=test_estimate,
    )
