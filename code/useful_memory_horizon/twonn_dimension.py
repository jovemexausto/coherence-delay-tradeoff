from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def twonn_dimension_estimate(sample: np.ndarray) -> float:
    if sample.ndim != 2:
        raise ValueError("sample must be a two-dimensional array")
    if sample.shape[0] < 3:
        raise ValueError("sample must contain at least three points")

    tree = cKDTree(sample)
    distances, _ = tree.query(sample, k=3)
    nearest = distances[:, 1]
    second = distances[:, 2]
    mask = np.isfinite(nearest) & np.isfinite(second) & (nearest > 0.0)
    ratios = second[mask] / nearest[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 1.0)]
    if ratios.size == 0:
        raise ValueError("TwoNN estimate is undefined for the provided sample")

    return float(ratios.size / np.sum(np.log(ratios)))
