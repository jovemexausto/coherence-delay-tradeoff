from __future__ import annotations

import numpy as np
from scipy.stats import f, fligner, levene, linregress


def durbin_watson(residuals: np.ndarray) -> float:
    residual_array = np.asarray(residuals, dtype=float)
    if residual_array.ndim != 1 or residual_array.size < 2:
        raise ValueError("residuals must be one-dimensional with size at least 2")
    denominator = float(np.sum(residual_array**2))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(np.diff(residual_array) ** 2))
    return numerator / denominator


def gaussian_kl_divergence(
    mean_p: float,
    var_p: float,
    mean_q: float,
    var_q: float,
    *,
    variance_floor: float = 1.0e-12,
) -> float:
    var_p = max(float(var_p), variance_floor)
    var_q = max(float(var_q), variance_floor)
    mean_p = float(mean_p)
    mean_q = float(mean_q)
    return 0.5 * (
        np.log(var_q / var_p) + (var_p + (mean_p - mean_q) ** 2) / var_q - 1.0
    )


def symmetrized_gaussian_kl_divergence(
    mean_p: float,
    var_p: float,
    mean_q: float,
    var_q: float,
    *,
    variance_floor: float = 1.0e-12,
) -> float:
    return 0.5 * (
        gaussian_kl_divergence(
            mean_p, var_p, mean_q, var_q, variance_floor=variance_floor
        )
        + gaussian_kl_divergence(
            mean_q, var_q, mean_p, var_p, variance_floor=variance_floor
        )
    )


def smoothed_histogram_kl_divergence(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
    pseudo_count: float = 0.5,
) -> float:
    reference_array = np.asarray(reference, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if reference_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("reference and target must be one-dimensional")
    if reference_array.size < 2 or target_array.size < 2:
        raise ValueError("reference and target must contain at least two values")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if pseudo_count < 0.0:
        raise ValueError("pseudo_count must be non-negative")

    combined = np.concatenate([reference_array, target_array])
    span = max(float(np.ptp(combined)), 1.0e-8)
    edges = np.linspace(
        float(np.min(combined)) - 0.05 * span,
        float(np.max(combined)) + 0.05 * span,
        bins + 1,
    )
    reference_counts, _ = np.histogram(reference_array, bins=edges)
    target_counts, _ = np.histogram(target_array, bins=edges)
    reference_probs = reference_counts.astype(float) + pseudo_count
    target_probs = target_counts.astype(float) + pseudo_count
    reference_probs /= float(np.sum(reference_probs))
    target_probs /= float(np.sum(target_probs))
    return float(np.sum(target_probs * np.log(target_probs / reference_probs)))


def symmetrized_smoothed_histogram_kl_divergence(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
    pseudo_count: float = 0.5,
) -> float:
    return 0.5 * (
        smoothed_histogram_kl_divergence(
            reference, target, bins=bins, pseudo_count=pseudo_count
        )
        + smoothed_histogram_kl_divergence(
            target, reference, bins=bins, pseudo_count=pseudo_count
        )
    )


def residual_scale_trend(
    residuals: np.ndarray,
    lags: np.ndarray,
    *,
    floor: float = 1.0e-8,
) -> np.ndarray:
    residual_array = np.asarray(residuals, dtype=float)
    lag_array = np.asarray(lags, dtype=float)
    if residual_array.ndim != 1 or lag_array.ndim != 1:
        raise ValueError("residuals and lags must be one-dimensional")
    if residual_array.shape != lag_array.shape:
        raise ValueError("residuals and lags must match")
    x = np.log(lag_array)
    y = np.log(np.abs(residual_array) + floor)
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    scale = np.exp(design @ beta)
    return np.maximum(scale, floor)


def standardized_residuals(
    residuals: np.ndarray,
    lags: np.ndarray,
    *,
    floor: float = 1.0e-8,
) -> np.ndarray:
    residual_array = np.asarray(residuals, dtype=float)
    return residual_array / residual_scale_trend(residual_array, lags, floor=floor)


def rolling_window_variances(
    values: np.ndarray,
    *,
    window_size: int,
    step: int,
) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size < 2:
        raise ValueError("values must be one-dimensional with size at least 2")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if step < 1:
        raise ValueError("step must be positive")
    if window_size > value_array.size:
        raise ValueError("window_size cannot exceed the series length")
    variances: list[float] = []
    for start in range(0, value_array.size - window_size + 1, step):
        window = value_array[start : start + window_size]
        variances.append(float(np.var(window, ddof=1)) if window.size > 1 else 0.0)
    return np.asarray(variances, dtype=float)


def window_centers(
    values: np.ndarray,
    *,
    window_size: int,
    step: int,
) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size < 2:
        raise ValueError("values must be one-dimensional with size at least 2")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if step < 1:
        raise ValueError("step must be positive")
    if window_size > value_array.size:
        raise ValueError("window_size cannot exceed the series length")
    centers: list[float] = []
    for start in range(0, value_array.size - window_size + 1, step):
        centers.append(float(start + 0.5 * (window_size - 1)))
    return np.asarray(centers, dtype=float)


def log_variance_trend(
    values: np.ndarray,
    lags: np.ndarray,
    *,
    window_size: int,
    step: int,
    floor: float = 1.0e-8,
) -> tuple[float, float]:
    value_array = np.asarray(values, dtype=float)
    lag_array = np.asarray(lags, dtype=float)
    if value_array.shape != lag_array.shape:
        raise ValueError("values and lags must match")
    variances = rolling_window_variances(
        value_array, window_size=window_size, step=step
    )
    centers = window_centers(lag_array, window_size=window_size, step=step)
    if variances.size < 2:
        return 0.0, 1.0
    result = linregress(np.log(centers), np.log(variances + floor))
    return float(result.slope), float(result.pvalue)


def cusum_squared_statistic(values: np.ndarray) -> float:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size < 2:
        raise ValueError("values must be one-dimensional with size at least 2")
    squared = value_array**2
    centered = squared - float(np.mean(squared))
    scale = float(np.std(centered, ddof=1))
    if scale <= 0.0:
        return 0.0
    cumulative = np.cumsum(centered) / (scale * np.sqrt(value_array.size))
    return float(np.max(np.abs(cumulative)))


def window_scale_test_p_values(
    values: np.ndarray,
    *,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size < 2:
        raise ValueError("values must be one-dimensional with size at least 2")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if step < 1:
        raise ValueError("step must be positive")
    if window_size > value_array.size:
        raise ValueError("window_size cannot exceed the series length")
    levene_p_values: list[float] = []
    fligner_p_values: list[float] = []
    for start in range(0, value_array.size - window_size + 1, step):
        window = value_array[start : start + window_size]
        complement = np.concatenate(
            [value_array[:start], value_array[start + window_size :]]
        )
        if complement.size < 2:
            break
        levene_p_values.append(
            float(levene(window, complement, center="median").pvalue)
        )
        fligner_p_values.append(float(fligner(window, complement).pvalue))
    return np.asarray(levene_p_values, dtype=float), np.asarray(
        fligner_p_values, dtype=float
    )


def sliding_window_kl_scores(
    values: np.ndarray,
    *,
    window_size: int,
    step: int,
    symmetrized: bool = True,
) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size < 2:
        raise ValueError("values must be one-dimensional with size at least 2")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if step < 1:
        raise ValueError("step must be positive")
    if window_size > value_array.size:
        raise ValueError("window_size cannot exceed the series length")

    bins = max(4, min(16, window_size // 2))
    scores: list[float] = []
    for start in range(0, value_array.size - window_size + 1, step):
        window = value_array[start : start + window_size]
        score = (
            symmetrized_smoothed_histogram_kl_divergence(
                value_array,
                window,
                bins=bins,
            )
            if symmetrized
            else smoothed_histogram_kl_divergence(
                value_array,
                window,
                bins=bins,
            )
        )
        scores.append(float(score))
    return np.asarray(scores, dtype=float)


def variance_window_kl_scores(
    values: np.ndarray,
    *,
    window_size: int,
    step: int,
    symmetrized: bool = True,
) -> np.ndarray:
    window_variances = rolling_window_variances(
        values, window_size=window_size, step=step
    )
    if window_variances.size < 2:
        return np.asarray([0.0], dtype=float)
    score_window_size = min(max(2, window_variances.size // 3), window_variances.size)
    score_step = 1
    return sliding_window_kl_scores(
        window_variances,
        window_size=score_window_size,
        step=score_step,
        symmetrized=symmetrized,
    )


def quadratic_curvature_p_value(
    log_observations: np.ndarray,
    lags: np.ndarray,
) -> float:
    y = np.asarray(log_observations, dtype=float)
    lag_array = np.asarray(lags, dtype=float)
    if y.shape != lag_array.shape:
        raise ValueError("log_observations and lags must match")
    x = np.log(lag_array)
    restricted = np.column_stack([np.ones_like(x), x])
    full = np.column_stack([np.ones_like(x), x, x**2])
    beta_restricted, *_ = np.linalg.lstsq(restricted, y, rcond=None)
    beta_full, *_ = np.linalg.lstsq(full, y, rcond=None)
    rss_restricted = float(np.sum((y - restricted @ beta_restricted) ** 2))
    rss_full = float(np.sum((y - full @ beta_full) ** 2))
    df_num = 1
    df_den = y.size - full.shape[1]
    if df_den <= 0:
        raise ValueError("need at least four lags for curvature test")
    improvement = max(rss_restricted - rss_full, 0.0)
    if rss_full <= 0.0:
        return 1.0 if improvement <= 0.0 else 0.0
    statistic = (improvement / float(df_num)) / (rss_full / float(df_den))
    return float(1.0 - f.cdf(statistic, df_num, df_den))


def dominant_periodogram_frequency(residuals: np.ndarray) -> float:
    residual_array = np.asarray(residuals, dtype=float)
    if residual_array.ndim != 1 or residual_array.size < 3:
        raise ValueError("residuals must be one-dimensional with size at least 3")
    centered = residual_array - float(np.mean(residual_array))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(centered.size, d=1.0)
    if spectrum.size <= 1:
        return 0.0
    peak_index = 1 + int(np.argmax(spectrum[1:]))
    return float(frequencies[peak_index])


def dominant_periodogram_power(residuals: np.ndarray) -> float:
    residual_array = np.asarray(residuals, dtype=float)
    if residual_array.ndim != 1 or residual_array.size < 3:
        raise ValueError("residuals must be one-dimensional with size at least 3")
    centered = residual_array - float(np.mean(residual_array))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    if spectrum.size <= 1:
        return 0.0
    return float(np.max(spectrum[1:]))
