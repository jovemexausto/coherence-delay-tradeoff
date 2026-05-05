"""Gaussian experiment package."""

from .model import (
    SampleComplexityResult,
    TGT_COLORS as GAUSSIAN_COLORS,
    TGT_CONDITIONS as GAUSSIAN_CONDITIONS,
    TGT_LABELS as GAUSSIAN_LABELS,
    TGTConfig as GaussianConfig,
    TGTResult as GaussianResult,
    UCurveResult,
    run_sample_complexity_experiment,
    run_tgt_ablation as run_gaussian_ablation,
    run_tgt_experiment as run_gaussian_experiment,
    run_ucurve_experiment,
)
