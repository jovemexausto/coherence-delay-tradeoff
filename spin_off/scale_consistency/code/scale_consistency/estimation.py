from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class RegressionEstimate:
    alpha: float
    H: float
    fitted: np.ndarray
    residuals: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class ScaleConsistencyTestResult:
    statistic: float
    critical_value: float
    reject: bool
    estimate: RegressionEstimate


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
    sigma0: float,
    n: int,
) -> RegressionEstimate:
    pilot = pilot_ols(log_observations, lags)
    weights = feasible_precision_weights(lags, pilot.alpha, pilot.H, sigma0, n)
    return weighted_least_squares(log_observations, lags, weights)


def residual_statistic(residuals: np.ndarray, weights: np.ndarray) -> float:
    residual_array = np.asarray(residuals, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return float(np.sum(weight_array * residual_array**2))


def run_scale_consistency_test(
    observed_discrepancies: np.ndarray,
    lags: np.ndarray,
    sigma0: float,
    n: int,
    *,
    alpha_level: float = 0.05,
) -> ScaleConsistencyTestResult:
    if not 0.0 < alpha_level < 1.0:
        raise ValueError("alpha_level must lie in (0, 1)")
    observed = np.asarray(observed_discrepancies, dtype=float)
    if np.any(observed <= 0.0):
        raise ValueError("observed discrepancies must be positive")
    estimate = feasible_wls(np.log(observed), np.asarray(lags, dtype=float), sigma0, n)
    statistic = residual_statistic(estimate.residuals, estimate.weights)
    df = observed.size - 2
    critical_value = float(chi2.ppf(1.0 - alpha_level, df=df))
    return ScaleConsistencyTestResult(
        statistic=statistic,
        critical_value=critical_value,
        reject=statistic > critical_value,
        estimate=estimate,
    )
