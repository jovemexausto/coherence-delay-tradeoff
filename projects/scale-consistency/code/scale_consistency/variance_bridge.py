from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .estimation import weighted_least_squares
from .horizon_bridge import continuous_optimal_horizon


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
class VarianceBridgeFit:
    model_kind: str
    alpha_pre: float
    H_pre: float
    alpha_post: float
    H_post: float
    zeta_pre: float
    zeta_post: float
    sigma_hat: np.ndarray
    variance_series: np.ndarray
    variance_r2: float
    change_point: float | None
    true_n_star: float
    n_star_pre: float
    n_star_post: float


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    window = max(3, min(window, values.size))
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _initial_fit(
    log_observations: np.ndarray, lags: np.ndarray
) -> tuple[float, float, np.ndarray]:
    y = np.asarray(log_observations, dtype=float)
    x = np.log(np.asarray(lags, dtype=float))
    weights = 1.0 / np.maximum(np.asarray(lags, dtype=float), 1.0)
    fit = weighted_least_squares(y, lags, weights)
    return fit.alpha, fit.H, np.asarray(fit.residuals, dtype=float)


def _smooth_local_variance(residuals: np.ndarray, window: int) -> np.ndarray:
    squared = np.asarray(residuals, dtype=float) ** 2
    smoothed = _moving_average(squared, window)
    return np.maximum(smoothed, 1.0e-10)


def _fit_variance_power_model(
    variance_series: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.log(np.asarray(variance_series, dtype=float) + 1.0e-12)
    x = np.asarray(centers, dtype=float)
    x_scaled = x / max(float(np.max(x)), 1.0)

    def residuals(params: np.ndarray) -> np.ndarray:
        log_sigma0, log_alpha, log_beta = params
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
        prediction = log_sigma0 + np.log1p(alpha * x_scaled**beta)
        return y - prediction

    p0 = np.array([float(np.median(y)), np.log(0.1), np.log(0.5)], dtype=float)
    result = least_squares(residuals, p0, method="trf")
    log_sigma0, log_alpha, log_beta = result.x
    alpha = float(np.exp(log_alpha))
    beta = float(np.exp(log_beta))
    sigma_sq = np.exp(log_sigma0) * (1.0 + alpha * x_scaled**beta)
    prediction = np.log(np.maximum(sigma_sq, 1.0e-12))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - prediction) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return np.sqrt(np.maximum(sigma_sq, 1.0e-12)), {
        "sigma0": float(np.exp(0.5 * log_sigma0)),
        "alpha": alpha,
        "beta": beta,
        "r2": r2,
    }


def _fit_variance_piecewise_model(
    variance_series: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.log(np.asarray(variance_series, dtype=float) + 1.0e-12)
    x = np.asarray(centers, dtype=float)
    n = y.size
    min_segment = max(3, n // 6)
    best = None
    best_sse = np.inf
    for split in range(min_segment, n - min_segment + 1):
        left = y[:split]
        right = y[split:]
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        sse = float(np.sum((left - left_mean) ** 2) + np.sum((right - right_mean) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = (split, left_mean, right_mean)
    if best is None:
        raise ValueError("could not fit piecewise variance model")
    split, left_mean, right_mean = best
    prediction = np.concatenate(
        [
            np.full(split, left_mean, dtype=float),
            np.full(n - split, right_mean, dtype=float),
        ]
    )
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - prediction) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    sigma_sq = np.exp(prediction)
    return np.sqrt(np.maximum(sigma_sq, 1.0e-12)), {
        "change_point": float(x[split - 1]),
        "left_sigma": float(np.exp(0.5 * left_mean)),
        "right_sigma": float(np.exp(0.5 * right_mean)),
        "r2": r2,
    }


def _fit_variance_smooth_model(
    variance_series: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.log(np.asarray(variance_series, dtype=float) + 1.0e-12)
    smoothed = _moving_average(y, max(5, y.size // 5))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - smoothed) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return np.sqrt(np.maximum(np.exp(smoothed), 1.0e-12)), {"r2": r2}


def fit_variance_model(
    observed_discrepancies: np.ndarray,
    lags: int | np.ndarray,
    *,
    sigma0: float | None,
    n: int,
    model_kind: str = "power",
    C_K: float = 1.0,
    C_S: float = 1.0,
    a: float = 0.5,
) -> VarianceBridgeFit:
    lag_array = _as_lag_array(lags)
    y = np.log(np.asarray(observed_discrepancies, dtype=float))
    alpha_pre, H_pre, residuals = _initial_fit(y, lag_array)
    local_window = max(5, min(21, lag_array.size // 4))
    variance_series = _smooth_local_variance(residuals, local_window)
    centers = lag_array[: variance_series.size]
    if model_kind == "piecewise":
        sigma_hat, variance_meta = _fit_variance_piecewise_model(
            variance_series, centers
        )
    elif model_kind == "smooth":
        sigma_hat, variance_meta = _fit_variance_smooth_model(variance_series, centers)
    else:
        sigma_hat, variance_meta = _fit_variance_power_model(variance_series, centers)
    if sigma_hat.size < lag_array.size:
        sigma_hat = np.pad(sigma_hat, (0, lag_array.size - sigma_hat.size), mode="edge")
    elif sigma_hat.size > lag_array.size:
        sigma_hat = sigma_hat[: lag_array.size]
    weights = 1.0 / np.maximum(sigma_hat**2, 1.0e-12)
    weighted = weighted_least_squares(y, lag_array, weights)
    alpha_post = float(weighted.alpha)
    H_post = float(weighted.H)
    zeta_pre = float(np.exp(alpha_pre))
    zeta_post = float(np.exp(alpha_post))
    true_n_star = continuous_optimal_horizon(C_K, a, C_S, zeta_post, H_post)
    return VarianceBridgeFit(
        model_kind=model_kind,
        alpha_pre=float(alpha_pre),
        H_pre=float(H_pre),
        alpha_post=alpha_post,
        H_post=H_post,
        zeta_pre=zeta_pre,
        zeta_post=zeta_post,
        sigma_hat=sigma_hat,
        variance_series=variance_series,
        variance_r2=float(variance_meta["r2"]),
        change_point=float(variance_meta.get("change_point", np.nan))
        if "change_point" in variance_meta
        else None,
        true_n_star=true_n_star,
        n_star_pre=continuous_optimal_horizon(C_K, a, C_S, zeta_pre, H_pre),
        n_star_post=true_n_star,
    )


def fit_best_variance_model(
    observed_discrepancies: np.ndarray,
    lags: int | np.ndarray,
    *,
    sigma0: float | None,
    n: int,
    C_K: float = 1.0,
    C_S: float = 1.0,
    a: float = 0.5,
) -> VarianceBridgeFit:
    fits = [
        fit_variance_model(
            observed_discrepancies,
            lags,
            sigma0=sigma0,
            n=n,
            model_kind=model_kind,
            C_K=C_K,
            C_S=C_S,
            a=a,
        )
        for model_kind in ("power", "piecewise", "smooth")
    ]
    return max(fits, key=lambda fit: fit.variance_r2)
