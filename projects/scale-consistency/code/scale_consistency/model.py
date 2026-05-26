from __future__ import annotations

from typing import Literal

import numpy as np


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


def exact_scale_profile(lags: int | np.ndarray, zeta: float, H: float) -> np.ndarray:
    lag_array = _as_lag_array(lags)
    if zeta <= 0.0:
        raise ValueError("zeta must be positive")
    return zeta * lag_array**H


def log_scale_signal(lags: int | np.ndarray, zeta: float, H: float) -> np.ndarray:
    lag_array = _as_lag_array(lags)
    return np.log(zeta) + H * np.log(lag_array)


def log_variance_profile(
    lags: int | np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
) -> np.ndarray:
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    if n <= 0:
        raise ValueError("n must be positive")
    D = exact_scale_profile(lags, zeta, H)
    return sigma0**2 / (float(n) * D**2)


def misspecified_scale_profile(
    lags: int | np.ndarray,
    zeta: float,
    H: float,
    amplitude: float,
    kind: Literal[
        "bump",
        "mixed",
        "piecewise",
        "sinusoid",
        "slope_shift",
    ] = "bump",
) -> np.ndarray:
    lag_array = _as_lag_array(lags)
    base = exact_scale_profile(lag_array, zeta, H)
    x = np.log(lag_array)
    if kind == "bump":
        center = 0.5 * (x.min() + x.max())
        width = max(0.35 * (x.max() - x.min()), 1.0e-8)
        perturbation = np.exp(-(((x - center) / width) ** 2))
    elif kind == "sinusoid":
        span = max(x.max() - x.min(), 1.0e-8)
        perturbation = np.sin(2.0 * np.pi * (x - x.min()) / span)
    elif kind == "slope_shift":
        midpoint = np.median(x)
        perturbation = np.where(x <= midpoint, -0.5, 0.5)
    elif kind == "piecewise":
        midpoint = np.median(x)
        left = lag_array <= np.exp(midpoint)
        profile = np.empty_like(base)
        profile[left] = zeta * lag_array[left] ** H
        if np.any(~left):
            continuity_scale = np.exp(-amplitude * midpoint)
            profile[~left] = (
                zeta * continuity_scale * lag_array[~left] ** (H + amplitude)
            )
        if np.any(profile <= 0.0):
            raise ValueError("misspecified profile became non-positive")
        return profile
    elif kind == "mixed":
        span = max(x.max() - x.min(), 1.0e-8)
        oscillation = np.sin(2.0 * np.pi * (x - x.min()) / span)
        jump_1 = np.where(lag_array >= np.quantile(lag_array, 0.35), 1.0, 0.0)
        jump_2 = np.where(lag_array >= np.quantile(lag_array, 0.7), 1.0, 0.0)
        profile = base * (1.0 + 0.5 * amplitude * oscillation)
        profile = profile * np.exp(0.5 * amplitude * jump_1 + amplitude * jump_2)
        if np.any(profile <= 0.0):
            raise ValueError("misspecified profile became non-positive")
        return profile
    else:
        raise ValueError(f"unsupported misspecification kind: {kind}")
    profile = base * (1.0 + amplitude * perturbation)
    if np.any(profile <= 0.0):
        raise ValueError("misspecified profile became non-positive")
    return profile


def simulate_log_observations(
    lags: int | np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
    *,
    kappa: float = 0.0,
    rng: np.random.Generator | None = None,
    profile: np.ndarray | None = None,
    noise: Literal[
        "gaussian",
        "bounded",
        "heteroskedastic",
        "heteroskedastic_ar",
        "heteroskedastic_power",
        "heteroskedastic_jump",
        "student",
    ] = "gaussian",
    student_df: float = 8.0,
    heteroskedastic_alpha: float = 0.0,
    heteroskedastic_beta: float = 1.0,
    heteroskedastic_jump_lag: float | None = None,
    heteroskedastic_rho: float = 0.0,
) -> np.ndarray:
    lag_array = _as_lag_array(lags)
    rng = np.random.default_rng() if rng is None else rng
    D = (
        exact_scale_profile(lag_array, zeta, H)
        if profile is None
        else np.asarray(profile, dtype=float)
    )
    if D.shape != lag_array.shape:
        raise ValueError("profile must match lag shape")
    if np.any(D <= 0.0):
        raise ValueError("profile must be positive")
    mean = np.log(D)
    variance = sigma0**2 / (float(n) * D**2)
    if noise in {
        "heteroskedastic",
        "heteroskedastic_ar",
        "heteroskedastic_power",
        "heteroskedastic_jump",
    }:
        x = np.log(lag_array)
        if noise == "heteroskedastic_ar":
            innovations = rng.normal(
                scale=abs(heteroskedastic_alpha), size=lag_array.size
            )
            state = np.zeros(lag_array.size, dtype=float)
            rho = float(np.clip(heteroskedastic_rho, -0.99, 0.99))
            for idx in range(1, lag_array.size):
                state[idx] = rho * state[idx - 1] + innovations[idx]
            scale = np.exp(state - float(np.mean(state)))
        elif noise == "heteroskedastic_jump":
            jump_lag = (
                float(np.median(lag_array))
                if heteroskedastic_jump_lag is None
                else float(heteroskedastic_jump_lag)
            )
            scale = 1.0 + abs(heteroskedastic_alpha) * (lag_array >= jump_lag)
        else:
            normalized_lag = lag_array / float(np.max(lag_array))
            scale = (
                1.0 + abs(heteroskedastic_alpha) * normalized_lag**heteroskedastic_beta
            )
        variance = variance * scale**2
    std = np.sqrt(variance)
    if noise == "gaussian":
        base_noise = rng.normal(size=lag_array.size)
    elif noise == "bounded":
        base_noise = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=lag_array.size)
    elif noise in {
        "heteroskedastic",
        "heteroskedastic_ar",
        "heteroskedastic_power",
        "heteroskedastic_jump",
    }:
        base_noise = rng.normal(size=lag_array.size)
    elif noise == "student":
        if student_df <= 2.0:
            raise ValueError("student_df must exceed 2")
        base_noise = rng.standard_t(df=student_df, size=lag_array.size)
        base_noise = base_noise / np.sqrt(student_df / (student_df - 2.0))
    else:
        raise ValueError(f"unsupported noise model: {noise}")
    inconsistency = (
        0.0
        if noise
        in {
            "heteroskedastic",
            "heteroskedastic_ar",
            "heteroskedastic_power",
            "heteroskedastic_jump",
        }
        else kappa * rng.normal(size=lag_array.size)
    )
    return mean + inconsistency + std * base_noise


def simulate_observed_discrepancies(
    lags: int | np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
    *,
    kappa: float = 0.0,
    rng: np.random.Generator | None = None,
    profile: np.ndarray | None = None,
    noise: Literal[
        "gaussian",
        "bounded",
        "heteroskedastic",
        "heteroskedastic_ar",
        "heteroskedastic_power",
        "heteroskedastic_jump",
        "student",
    ] = "gaussian",
    student_df: float = 8.0,
    heteroskedastic_alpha: float = 0.0,
    heteroskedastic_beta: float = 1.0,
    heteroskedastic_jump_lag: float | None = None,
    heteroskedastic_rho: float = 0.0,
) -> np.ndarray:
    return np.exp(
        simulate_log_observations(
            lags,
            zeta,
            H,
            sigma0,
            n,
            kappa=kappa,
            rng=rng,
            profile=profile,
            noise=noise,
            student_df=student_df,
            heteroskedastic_alpha=heteroskedastic_alpha,
            heteroskedastic_beta=heteroskedastic_beta,
            heteroskedastic_jump_lag=heteroskedastic_jump_lag,
            heteroskedastic_rho=heteroskedastic_rho,
        )
    )
