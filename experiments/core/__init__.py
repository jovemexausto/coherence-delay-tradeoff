"""Shared experiment helpers."""

# pyright: reportUnusedImport=false

from .common import (
    export_summary_csv,
    format_summary_markdown,
    OnsetSummary,
    match_warnings_to_events,
    rolling_mean,
    WarningMatchResult,
    summarize_onset,
    threshold_crossings,
)
from .baselines import (
    ScalarDetectionResult,
    run_cusum_detector,
    run_forgetting_factor_rls_detector,
    run_frechet_detector,
    run_scalar_kalman_detector,
)
from .harness import ExperimentHarness
from .detectors import run_river_drift_detector
from .regime_map import save_regime_first_summary_figure
from .sinkhorn import SinkhornResult, debiased_sinkhorn_divergence, sinkhorn_cost
from .types import SummaryRow, SummaryRows, SummaryValue
