from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from temporalbridge._backend import fit_lag_power_law


def _as_lag_array(lags: np.ndarray | list[float]) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    if lag_array.ndim != 1 or lag_array.size < 2:
        raise ValueError("lags must be a one-dimensional array with size at least 2")
    if np.any(lag_array <= 0.0):
        raise ValueError("lags must be positive")
    return lag_array


def fit_horizon(
    lags: np.ndarray,
    discrepancies: np.ndarray,
    method: str = "power_law",
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the temporal-validity horizon from lag geometry.

    The current spin-off surface supports the validated power-law lag model from
    the sibling `scale_consistency` project and returns plain Python/numpy
    structures so adapters can remain thin.
    """

    if method != "power_law":
        raise ValueError(f"unsupported fit method: {method}")

    lag_array = _as_lag_array(lags)
    discrepancy_array = np.asarray(discrepancies, dtype=float)
    if discrepancy_array.shape != lag_array.shape:
        raise ValueError("discrepancies must match lag shape")
    if np.any(discrepancy_array <= 0.0):
        raise ValueError("discrepancies must be positive")

    options = dict(fit_options or {})
    sigma0 = float(options.get("sigma0", 1.0))
    n = int(options.get("n", 1000))
    C_K = float(options.get("C_K", 1.0))
    C_S = float(options.get("C_S", 1.0))
    a = float(options.get("a", 0.5))

    estimate = fit_lag_power_law(discrepancy_array, lag_array, sigma0=sigma0, n=n)
    loss = float(np.sum(estimate.weights * estimate.residuals**2))
    clipped_H = max(float(estimate.H), 1.0e-6)
    clipped_zeta = max(float(estimate.zeta), 1.0e-12)
    return {
        "H": float(estimate.H),
        "zeta": float(estimate.zeta),
        "n_star": float(
            (a * C_K / (clipped_H * C_S * clipped_zeta)) ** (1.0 / (a + clipped_H))
        ),
        "profile": {
            "lags": lag_array.copy(),
            "D_j": discrepancy_array.copy(),
        },
        "fit_stats": {
            "loss": loss,
            "fitted_log": np.asarray(estimate.fitted, dtype=float).copy(),
            "residuals": np.asarray(estimate.residuals, dtype=float).copy(),
            "weights": np.asarray(estimate.weights, dtype=float).copy(),
        },
        "fit_options": {
            "sigma0": sigma0,
            "n": n,
            "C_K": C_K,
            "C_S": C_S,
            "a": a,
            "method": method,
        },
    }
