from __future__ import annotations

import numpy as np

from .model import RajputResult


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan"), float("nan")
    err = y_true[valid] - y_pred[valid]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse


def build_rajput_rows(result: RajputResult) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    y_true = result.targets[result.test_slice]
    cap_test = result.caps[result.test_slice]
    cap_fraction = (
        float(np.mean(cap_test < result.buffer_sizes.max())) if cap_test.size else 0.0
    )
    cap_mean = float(np.mean(cap_test)) if cap_test.size else float("nan")

    methods = (
        ("single", result.single_mean, result.single_std),
        ("naive_ensemble", result.naive_mean, result.naive_std),
        ("uq_ensemble", result.uq_mean, result.uq_std),
        ("uq_ensemble_umr", result.uq_umr_mean, result.uq_umr_std),
    )

    best_buffer = int(result.buffer_sizes[result.single_index])
    for name, mean, std in methods:
        mae, rmse = _metrics(y_true, mean[result.test_slice])
        std_slice = std[result.test_slice]
        valid = (
            np.isfinite(y_true)
            & np.isfinite(mean[result.test_slice])
            & np.isfinite(std_slice)
        )
        calib = (
            float(
                np.mean(
                    np.abs(y_true[valid] - mean[result.test_slice][valid])
                    / np.maximum(std_slice[valid], 1e-9)
                )
            )
            if np.any(valid)
            else float("nan")
        )
        rows.append(
            {
                "dataset": result.config.dataset,
                "method": name,
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "calibration_proxy": round(calib, 6),
                "best_single_buffer": best_buffer,
                "mean_cap": round(cap_mean, 2),
                "cap_fraction": round(cap_fraction, 3),
                "single_buffer_calibration_mae": round(
                    result.calibrations[result.single_index].calibration_mae, 6
                ),
                "single_buffer_alpha": round(
                    result.calibrations[result.single_index].alpha, 4
                ),
            }
        )
    return rows
