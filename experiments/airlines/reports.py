from __future__ import annotations

import numpy as np

from .model import AirlinesBenchmarkResult


def _summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_prob)
    if not np.any(valid):
        return {"accuracy": 0.0, "log_loss": float("nan"), "brier": float("nan")}
    y_true = y_true[valid].astype(float)
    y_prob = y_prob[valid].astype(float)
    y_pred = (y_prob >= 0.5).astype(float)
    accuracy = float(np.mean(y_pred == y_true))
    log_loss = float(
        -np.mean(
            y_true * np.log(np.clip(y_prob, 1e-9, 1.0 - 1e-9))
            + (1.0 - y_true) * np.log(np.clip(1.0 - y_prob, 1e-9, 1.0))
        )
    )
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {"accuracy": accuracy, "log_loss": log_loss, "brier": brier}


def _cap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    caps: np.ndarray,
    *,
    threshold: int,
) -> dict[str, float]:
    mask = np.isfinite(y_prob) & (caps <= float(threshold))
    if not np.any(mask):
        return {"accuracy": 0.0, "log_loss": float("nan"), "share": 0.0}
    summary = _summarize(y_true[mask], y_prob[mask])
    summary["share"] = float(np.mean(caps <= float(threshold)))
    return summary


def build_airlines_rows(
    result: AirlinesBenchmarkResult,
) -> list[dict[str, float | int | str]]:
    test_slice = result.test_slice
    y_true = result.targets[test_slice]
    caps = result.caps[test_slice]
    rows: list[dict[str, float | int | str]] = []

    strategies = [
        ("raw", result.base_probabilities, None),
        ("single", result.single_probabilities, None),
        ("single_cap", result.single_cap_probabilities, "single"),
        ("ensemble", result.ensemble_probabilities, None),
        ("ensemble_cap", result.ensemble_cap_probabilities, "ensemble"),
    ]

    summaries: dict[str, dict[str, float]] = {}
    for name, probabilities, _ in strategies:
        summaries[name] = _summarize(y_true, probabilities[test_slice])

    for name, probabilities, counterpart in strategies:
        summary = summaries[name]
        cap_threshold = (
            result.selected_window
            if "single" in name
            else int(result.window_sizes.max())
        )
        cap_summary = _cap_metrics(
            y_true,
            probabilities[test_slice],
            caps,
            threshold=cap_threshold,
        )
        delta_accuracy = 0.0
        delta_log_loss = 0.0
        if counterpart is not None:
            delta_accuracy = round(
                summary["accuracy"] - summaries[counterpart]["accuracy"], 4
            )
            delta_log_loss = round(
                summary["log_loss"] - summaries[counterpart]["log_loss"], 4
            )
        rows.append(
            {
                "strategy": name,
                "window": int(
                    result.selected_window
                    if "single" in name
                    else result.window_sizes.max()
                ),
                "accuracy": round(summary["accuracy"], 4),
                "log_loss": round(summary["log_loss"], 4),
                "brier": round(summary["brier"], 4),
                "cap_accuracy": round(cap_summary["accuracy"], 4),
                "cap_log_loss": round(cap_summary["log_loss"], 4),
                "cap_share": round(cap_summary["share"], 4),
                "delta_accuracy_vs_counterpart": delta_accuracy,
                "delta_log_loss_vs_counterpart": delta_log_loss,
                "cap_mean": round(float(np.mean(caps)), 2),
                "cap_median": round(float(np.median(caps)), 2),
                "selected_window": int(result.selected_window),
            }
        )
    return rows
