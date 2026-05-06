from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .common import match_warnings_to_events


@dataclass(slots=True)
class ScalarDetectionResult:
    sigma: np.ndarray
    estimate: np.ndarray
    warnings: list[int]
    matched_warnings: list[int]
    matched_events: list[int]
    lead_times: list[int]


def _prefix_stats(values: np.ndarray, prefix_length: int) -> tuple[int, float, float]:
    prefix_length = max(1, min(int(prefix_length), values.size))
    prefix = values[:prefix_length]
    mean = float(np.mean(prefix))
    std = float(np.std(prefix))
    if not np.isfinite(std) or std <= 1e-9:
        std = 1.0
    return prefix_length, mean, std


def _package_result(
    sigma: np.ndarray,
    estimate: np.ndarray,
    warnings: list[int],
    events: list[int],
    max_gap: int,
) -> ScalarDetectionResult:
    match_result = match_warnings_to_events(warnings, events, max_gap)
    return ScalarDetectionResult(
        sigma=sigma,
        estimate=estimate,
        warnings=warnings,
        matched_warnings=match_result.matched_warnings,
        matched_events=match_result.matched_events,
        lead_times=match_result.lead_times,
    )


def _rolling_window_stats(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, min(int(window), values.size))
    csum = np.cumsum(np.insert(values, 0, 0.0))
    csum2 = np.cumsum(np.insert(values * values, 0, 0.0))
    totals = csum[window:] - csum[:-window]
    totals2 = csum2[window:] - csum2[:-window]
    means = totals / window
    variances = np.maximum(totals2 / window - means * means, 0.0)
    stds = np.sqrt(variances)
    return means, stds


def run_cusum_detector(
    values: np.ndarray,
    events: list[int],
    max_gap: int,
    *,
    warning_threshold: float,
    prefix_length: int = 2000,
    drift_allowance: float = 0.25,
    alarm_scale: float = 8.0,
) -> ScalarDetectionResult:
    prefix_length, baseline_mean, baseline_std = _prefix_stats(values, prefix_length)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, baseline_mean)
    warnings: list[int] = []
    below = False
    pos_sum = 0.0
    neg_sum = 0.0
    alarm_level = max(alarm_scale * baseline_std, 1e-6)

    for index, value in enumerate(values):
        if index < prefix_length:
            sigma[index] = 1.0
            estimate[index] = baseline_mean
            continue

        residual = float(value - baseline_mean)
        pos_sum = max(0.0, pos_sum + residual - drift_allowance * baseline_std)
        neg_sum = max(0.0, neg_sum - residual - drift_allowance * baseline_std)
        stat = max(pos_sum, neg_sum)
        estimate[index] = baseline_mean
        sigma[index] = 1.0 / (1.0 + stat / alarm_level)
        if sigma[index] < warning_threshold and not below:
            warnings.append(index)
            below = True
            pos_sum = 0.0
            neg_sum = 0.0
        elif sigma[index] >= warning_threshold:
            below = False

    return _package_result(sigma, estimate, warnings, events, max_gap)


def run_forgetting_factor_rls_detector(
    values: np.ndarray,
    events: list[int],
    max_gap: int,
    *,
    warning_threshold: float,
    prefix_length: int = 2000,
    forgetting_factor: float = 0.995,
) -> ScalarDetectionResult:
    _, theta, baseline_std = _prefix_stats(values, prefix_length)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, theta)
    warnings: list[int] = []
    below = False
    covariance = baseline_std**2

    for index, value in enumerate(values):
        if index < prefix_length:
            sigma[index] = 1.0
            estimate[index] = theta
            continue

        prediction = theta
        innovation = float(value - prediction)
        innovation_scale = max(math.sqrt(covariance + baseline_std**2), baseline_std)
        sigma[index] = 1.0 / (1.0 + 0.5 * (innovation / innovation_scale) ** 2)
        estimate[index] = prediction
        if sigma[index] < warning_threshold and not below:
            warnings.append(index)
            below = True
        elif sigma[index] >= warning_threshold:
            below = False

        gain = covariance / (forgetting_factor + covariance)
        theta = theta + gain * innovation
        covariance = max((1.0 - gain) * covariance / forgetting_factor, 1e-9)

    return _package_result(sigma, estimate, warnings, events, max_gap)


def run_scalar_kalman_detector(
    values: np.ndarray,
    events: list[int],
    max_gap: int,
    *,
    warning_threshold: float,
    prefix_length: int = 2000,
    process_scale: float = 0.02,
) -> ScalarDetectionResult:
    _, theta, baseline_std = _prefix_stats(values, prefix_length)
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, theta)
    warnings: list[int] = []
    below = False
    process_var = (process_scale * baseline_std) ** 2
    measurement_var = baseline_std**2
    covariance = baseline_std**2

    for index, value in enumerate(values):
        if index < prefix_length:
            sigma[index] = 1.0
            estimate[index] = theta
            continue

        prediction = theta
        predicted_covariance = covariance + process_var
        innovation = float(value - prediction)
        innovation_scale = math.sqrt(predicted_covariance + measurement_var)
        sigma[index] = 1.0 / (1.0 + 0.5 * (innovation / innovation_scale) ** 2)
        estimate[index] = prediction
        if sigma[index] < warning_threshold and not below:
            warnings.append(index)
            below = True
        elif sigma[index] >= warning_threshold:
            below = False

        gain = predicted_covariance / (predicted_covariance + measurement_var)
        theta = prediction + gain * innovation
        covariance = max((1.0 - gain) * predicted_covariance, 1e-9)

    return _package_result(sigma, estimate, warnings, events, max_gap)


def run_frechet_detector(
    values: np.ndarray,
    events: list[int],
    max_gap: int,
    *,
    warning_threshold: float,
    prefix_length: int = 2000,
    window_size: int = 100,
) -> ScalarDetectionResult:
    _, baseline_mean, baseline_std = _prefix_stats(values, prefix_length)
    window_size = max(1, min(int(window_size), values.size))
    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, baseline_mean)
    warnings: list[int] = []
    below = False
    reference_scale = max(baseline_std, 1e-6)
    means, stds = _rolling_window_stats(values, window_size)

    for offset, (current_mean, current_std) in enumerate(zip(means, stds)):
        index = offset + window_size - 1
        distance = math.sqrt(
            (float(current_mean) - baseline_mean) ** 2
            + (float(current_std) - baseline_std) ** 2
        )
        sigma[index] = 1.0 / (1.0 + distance / reference_scale)
        estimate[index] = float(current_mean)
        if index < prefix_length:
            sigma[index] = 1.0
            continue
        if sigma[index] < warning_threshold and not below:
            warnings.append(index)
            below = True
        elif sigma[index] >= warning_threshold:
            below = False

    return _package_result(sigma, estimate, warnings, events, max_gap)


def _rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF kernel matrix between 1-D sample vectors."""
    diff = x[:, None] - y[None, :]
    return np.exp(-diff**2 / (2.0 * bandwidth**2))


def _unbiased_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Unbiased estimate of MMD^2 with Gaussian RBF kernel."""
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        return 0.0
    kxx = _rbf_kernel_matrix(x, x, bandwidth)
    kyy = _rbf_kernel_matrix(y, y, bandwidth)
    kxy = _rbf_kernel_matrix(x, y, bandwidth)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_xx = kxx.sum() / (m * (m - 1))
    term_yy = kyy.sum() / (n * (n - 1))
    term_xy = kxy.sum() / (m * n)
    return float(term_xx + term_yy - 2.0 * term_xy)


def run_mmd_detector(
    values: np.ndarray,
    events: list[int],
    max_gap: int,
    *,
    warning_threshold: float,
    prefix_length: int = 2000,
    window_size: int = 100,
) -> ScalarDetectionResult:
    """Kernel MMD two-sample test baseline detector.

    Compares each sliding window against a fixed-size subsample of the
    healthy prefix using an unbiased estimate of the squared MMD with a
    Gaussian RBF kernel.  Bandwidth is set via the median heuristic on the
    prefix.  The reference is subsampled to ``window_size`` points and its
    self-kernel is precomputed so that the per-step cost is O(window_size^2)
    rather than O(prefix_length^2).  A stride of ``window_size // 4``
    reduces the total number of evaluations on long streams; intermediate
    steps carry forward the previous score.
    """
    prefix_length = max(1, min(int(prefix_length), values.size))
    window_size = max(1, min(int(window_size), values.size))
    reference_full = values[:prefix_length].copy()

    # Median heuristic for bandwidth (on a subsample to keep it fast)
    sub_bw = reference_full[:: max(1, prefix_length // 500)]
    pairwise_dists = np.abs(sub_bw[:, None] - sub_bw[None, :])
    bandwidth = max(float(np.median(pairwise_dists[pairwise_dists > 0])), 1e-6)

    # Fixed-size reference subsample for all comparisons
    rng = np.random.RandomState(42)
    ref_idx = rng.choice(prefix_length, size=min(window_size, prefix_length), replace=False)
    reference = reference_full[ref_idx]
    ref_n = reference.shape[0]

    # Precompute reference self-kernel term
    k_ref = _rbf_kernel_matrix(reference, reference, bandwidth)
    np.fill_diagonal(k_ref, 0.0)
    ref_self_term = k_ref.sum() / (ref_n * (ref_n - 1)) if ref_n > 1 else 0.0

    sigma = np.full(values.size, np.nan)
    estimate = np.full(values.size, float(np.mean(reference_full)))
    warnings: list[int] = []
    below = False

    # Null scale from prefix splits
    null_mmds: list[float] = []
    for _ in range(min(20, prefix_length // window_size)):
        idx_a = rng.choice(prefix_length, size=window_size, replace=False)
        idx_b = rng.choice(prefix_length, size=window_size, replace=False)
        null_mmds.append(max(_unbiased_mmd2(reference_full[idx_a], reference_full[idx_b], bandwidth), 0.0))
    null_scale = max(float(np.mean(null_mmds)) if null_mmds else 1e-6, 1e-6)

    stride = max(1, window_size // 4)
    last_sigma = 1.0

    for index in range(window_size - 1, values.size):
        if index < prefix_length:
            sigma[index] = 1.0
            estimate[index] = float(np.mean(values[index - window_size + 1 : index + 1]))
            continue

        if (index - prefix_length) % stride != 0:
            sigma[index] = last_sigma
            estimate[index] = float(np.mean(values[index - window_size + 1 : index + 1]))
        else:
            current_window = values[index - window_size + 1 : index + 1]
            m = current_window.shape[0]
            # Compute only the window self-kernel and cross-kernel
            k_ww = _rbf_kernel_matrix(current_window, current_window, bandwidth)
            np.fill_diagonal(k_ww, 0.0)
            ww_term = k_ww.sum() / (m * (m - 1)) if m > 1 else 0.0
            k_wr = _rbf_kernel_matrix(current_window, reference, bandwidth)
            cross_term = k_wr.sum() / (m * ref_n)
            mmd2 = max(ww_term + ref_self_term - 2.0 * cross_term, 0.0)
            last_sigma = 1.0 / (1.0 + mmd2 / null_scale)
            sigma[index] = last_sigma
            estimate[index] = float(np.mean(current_window))

        if sigma[index] < warning_threshold and not below:
            warnings.append(index)
            below = True
        elif sigma[index] >= warning_threshold:
            below = False

    return _package_result(sigma, estimate, warnings, events, max_gap)
