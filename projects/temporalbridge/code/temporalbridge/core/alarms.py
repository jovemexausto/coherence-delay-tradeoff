from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def calibrate_alarms(
    profile: Mapping[str, Any],
    bootstrap_results: Mapping[str, Any],
    diagnostics: Mapping[str, Sequence[float]],
    quantiles: Sequence[float] = (0.95, 0.99),
) -> dict[str, Any]:
    """Produce empirical diagnostic thresholds from bootstrap/null summaries.

    In this initial spin-off version, if no explicit bootstrap null series are
    supplied, the observed diagnostic series acts as a fallback empirical proxy.
    """

    thresholds: dict[str, dict[str, float]] = {}
    null_source = bootstrap_results.get("diagnostic_bootstrap", diagnostics)
    for name, values in diagnostics.items():
        empirical = np.asarray(null_source.get(name, values), dtype=float)
        thresholds[name] = {
            f"q{int(100 * q)}": float(np.quantile(empirical, q)) for q in quantiles
        }
    return {
        "profile": dict(profile),
        "bootstrap_method": bootstrap_results.get("method", "unknown"),
        "thresholds": thresholds,
        "notes": "wild default; use moving_block when scale dependence is detected",
    }


def detect_alarms(
    diagnostics: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, Mapping[str, float]],
    persistence_windows: int = 1,
) -> dict[str, Any]:
    """Apply calibrated thresholds and return alarm traces."""

    if persistence_windows <= 0:
        raise ValueError("persistence_windows must be positive")
    alarms: dict[str, list[int]] = {}
    summary: dict[str, dict[str, int | None]] = {}
    for name, values in diagnostics.items():
        series = np.asarray(values, dtype=float)
        threshold = float(next(iter(thresholds[name].values())))
        exceed = series > threshold
        hits: list[int] = []
        run = 0
        for idx, flag in enumerate(exceed):
            run = run + 1 if flag else 0
            if run >= persistence_windows:
                hits.append(idx)
        alarms[name] = hits
        summary[name] = {
            "first_alarm_time": hits[0] if hits else None,
            "alarm_count": len(hits),
        }
    return {
        "alarms": alarms,
        "alarm_summary": summary,
    }
