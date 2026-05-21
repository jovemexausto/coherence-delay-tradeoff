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


def kappa_boundary(n: int, L: int, constant: float = 1.0) -> float:
    if n <= 0 or L <= 0:
        raise ValueError("n and L must be positive")
    return float(constant) / math.sqrt(float(n * L))


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
    if n <= 0 or L <= 0:
        raise ValueError("n and L must be positive")
    return float(rmse) * math.sqrt(float(n) * float(L) ** (2.0 * H + 1.0))
