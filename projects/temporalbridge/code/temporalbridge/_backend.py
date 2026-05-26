from __future__ import annotations


try:
    from scale_consistency.horizon_bridge import (  # type: ignore
        bootstrap_lag_power_law,
        continuous_optimal_horizon,
        fit_lag_power_law,
        plug_in_horizon,
    )
    from scale_consistency.bridge_diagnostics import (  # type: ignore
        durbin_watson,
        quadratic_curvature_p_value,
        sliding_window_kl_scores,
        standardized_residuals,
    )
    from scale_consistency.model import simulate_observed_discrepancies  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised through import path setup
    raise ImportError(
        "temporalbridge currently depends on the sibling project "
        "'projects/scale-consistency/code'. Add that directory to PYTHONPATH."
    ) from exc


__all__ = [
    "bootstrap_lag_power_law",
    "continuous_optimal_horizon",
    "durbin_watson",
    "fit_lag_power_law",
    "plug_in_horizon",
    "quadratic_curvature_p_value",
    "simulate_observed_discrepancies",
    "sliding_window_kl_scores",
    "standardized_residuals",
]
