from __future__ import annotations

import numpy as np


def profile_from_arrays(
    lags: np.ndarray, discrepancies: np.ndarray
) -> dict[str, np.ndarray]:
    lag_array = np.asarray(lags, dtype=float)
    discrepancy_array = np.asarray(discrepancies, dtype=float)
    if lag_array.shape != discrepancy_array.shape:
        raise ValueError("lags and discrepancies must match")
    return {
        "lags": lag_array,
        "D_j": discrepancy_array,
    }
