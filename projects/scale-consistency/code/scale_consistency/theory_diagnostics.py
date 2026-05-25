from __future__ import annotations

import math

import numpy as np

from .estimation import design_matrix, oracle_precision_weights


def chi_square_degrees_of_freedom(lags: int | np.ndarray) -> int:
    size = int(lags) if isinstance(lags, int) else int(np.asarray(lags).size)
    if size < 3:
        raise ValueError("need at least three lags")
    return size - 2


def chi_square_null_mean(lags: int | np.ndarray) -> float:
    return float(chi_square_degrees_of_freedom(lags))


def chi_square_null_variance(lags: int | np.ndarray) -> float:
    return 2.0 * float(chi_square_degrees_of_freedom(lags))


def lag_energy(lags: np.ndarray, H: float) -> float:
    lag_array = np.asarray(lags, dtype=float)
    if lag_array.ndim != 1 or lag_array.size == 0:
        raise ValueError("lags must be a non-empty one-dimensional array")
    if np.any(lag_array <= 0.0):
        raise ValueError("lags must be positive")
    return float(np.sum(lag_array ** (2.0 * float(H))))


def information_scale(n: int, lags: np.ndarray, H: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    return float(n) * lag_energy(lags, H)


def kappa_boundary(
    n: int,
    lags: np.ndarray,
    H: float,
    constant: float = 1.0,
) -> float:
    return float(constant) / math.sqrt(information_scale(n, lags, H))


def oracle_h_variance(
    lags: np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
) -> float:
    lag_array = np.asarray(lags, dtype=float)
    X = design_matrix(lag_array)
    W = np.diag(oracle_precision_weights(lag_array, zeta, H, sigma0, n))
    covariance = np.linalg.inv(X.T @ W @ X)
    return float(covariance[1, 1])


def scaled_rmse_constant(rmse: float, n: int, L: int, H: float) -> float:
    if rmse < 0.0:
        raise ValueError("rmse must be non-negative")
    lags = np.arange(1, int(L) + 1, dtype=float)
    return float(rmse) * math.sqrt(information_scale(n, lags, H))
