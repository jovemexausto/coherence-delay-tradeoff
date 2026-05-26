from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from temporalbridge._backend import (
    durbin_watson,
    quadratic_curvature_p_value,
    sliding_window_kl_scores,
    standardized_residuals,
)


def compute_profile_diagnostics(
    profile: Mapping[str, Any],
) -> dict[str, np.ndarray | float]:
    lags = np.asarray(profile["profile"]["lags"], dtype=float)
    residuals = np.asarray(profile["fit_stats"]["residuals"], dtype=float)
    observed = np.asarray(profile["profile"]["D_j"], dtype=float)
    window_size = max(4, min(16, len(lags) // 3))
    step = max(1, window_size // 3)
    return {
        "KL_residual": sliding_window_kl_scores(
            residuals, window_size=window_size, step=step
        ),
        "KL_standardized": sliding_window_kl_scores(
            standardized_residuals(residuals, lags),
            window_size=window_size,
            step=step,
        ),
        "DW": float(durbin_watson(residuals)),
        "curvature_p": float(quadratic_curvature_p_value(np.log(observed), lags)),
    }
