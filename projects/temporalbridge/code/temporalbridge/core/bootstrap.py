from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from temporalbridge._backend import (
    fit_lag_power_law,
    simulate_observed_discrepancies,
)


def _wild_bootstrap_log_observations(
    *, fitted: np.ndarray, residuals: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    multipliers = rng.choice(np.array([-1.0, 1.0], dtype=float), size=residuals.size)
    return fitted + residuals * multipliers


def _moving_block_bootstrap_log_observations(
    *,
    fitted: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    residual_array = np.asarray(residuals, dtype=float)
    n_obs = residual_array.size
    effective_block_length = max(2, min(int(block_length), n_obs))
    starts = np.arange(0, n_obs - effective_block_length + 1)
    centered = residual_array - float(np.mean(residual_array))
    sampled: list[np.ndarray] = []
    while sum(block.size for block in sampled) < n_obs:
        start = int(rng.choice(starts))
        sampled.append(centered[start : start + effective_block_length])
    return fitted + np.concatenate(sampled)[:n_obs]


def _percentile_interval(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = 0.5 * (1.0 - ci)
    lower, upper = np.quantile(np.asarray(samples, dtype=float), (alpha, 1.0 - alpha))
    return float(lower), float(upper)


def bootstrap_horizon(
    profile: Mapping[str, Any],
    method: str = "wild",
    n_boot: int = 1000,
    block_length: int | None = None,
    ci: float = 0.95,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap inference for the plug-in horizon.

    The input `profile` is expected to be the output of `fit_horizon`.
    """

    if method not in {"parametric", "wild", "moving_block"}:
        raise ValueError(f"unsupported bootstrap method: {method}")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie in (0, 1)")

    lags = np.asarray(profile["profile"]["lags"], dtype=float)
    discrepancies = np.asarray(profile["profile"]["D_j"], dtype=float)
    options = dict(profile.get("fit_options", {}))
    sigma0 = float(options.get("sigma0", 1.0))
    n = int(options.get("n", 1000))
    C_K = float(options.get("C_K", 1.0))
    C_S = float(options.get("C_S", 1.0))
    a = float(options.get("a", 0.5))

    estimate = fit_lag_power_law(discrepancies, lags, sigma0=sigma0, n=n)
    rng = np.random.default_rng(rng_seed)
    H_samples = np.empty(n_boot, dtype=float)
    n_star_samples = np.empty(n_boot, dtype=float)
    effective_block_length = (
        max(3, int(0.1 * len(lags))) if block_length is None else int(block_length)
    )

    for idx in range(n_boot):
        if method == "parametric":
            bootstrap_obs = simulate_observed_discrepancies(
                lags,
                estimate.zeta,
                estimate.H,
                sigma0,
                n,
                rng=rng,
            )
        elif method == "wild":
            bootstrap_obs = np.exp(
                _wild_bootstrap_log_observations(
                    fitted=np.asarray(estimate.fitted, dtype=float),
                    residuals=np.asarray(estimate.residuals, dtype=float),
                    rng=rng,
                )
            )
        else:
            bootstrap_obs = np.exp(
                _moving_block_bootstrap_log_observations(
                    fitted=np.asarray(estimate.fitted, dtype=float),
                    residuals=np.asarray(estimate.residuals, dtype=float),
                    rng=rng,
                    block_length=effective_block_length,
                )
            )
        bootstrap_estimate = fit_lag_power_law(bootstrap_obs, lags, sigma0=sigma0, n=n)
        H_samples[idx] = bootstrap_estimate.H
        clipped_H = max(float(bootstrap_estimate.H), 1.0e-6)
        clipped_zeta = max(float(bootstrap_estimate.zeta), 1.0e-12)
        n_star_samples[idx] = float(
            (a * C_K / (clipped_H * C_S * clipped_zeta)) ** (1.0 / (a + clipped_H))
        )

    return {
        "method": method,
        "ci_H": _percentile_interval(H_samples, ci),
        "ci_n_star": _percentile_interval(n_star_samples, ci),
        "boot_dist_H": H_samples,
        "boot_dist_n_star": n_star_samples,
        "metadata": {
            "n_boot": n_boot,
            "ci": ci,
            "block_length": effective_block_length
            if method == "moving_block"
            else None,
        },
    }
