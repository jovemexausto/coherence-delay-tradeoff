from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidityState:
    n_star: float
    ci_n_star: tuple[float, float]
    H: float
    ci_H: tuple[float, float]
    identifiability_score: float
    diagnostics: dict[str, float]
    diagnostic_thresholds: dict[str, float]
    scale_dependence_detected: bool = False
    alarm_persistence: int = 0
    bootstrap_mode: str = "wild"


@dataclass(frozen=True)
class ControllerParams:
    deploy_width_fraction_n_star: float = 0.02
    deploy_width_fraction_H: float = 0.03
    band_width_fraction_n_star: float = 0.10
    band_width_fraction_H: float = 0.04
    S_min: float = 0.50
    S_min_band: float = 0.35
    persistence_required: int = 1
    alarm_min_diagnostics: int = 2


@dataclass(frozen=True)
class ControllerDecision:
    action: Literal["use_n_star", "use_band", "hold", "alarm"]
    reason: str


def validity_controller(
    state: ValidityState,
    params: ControllerParams = ControllerParams(),
) -> ControllerDecision:
    n_width = float(state.ci_n_star[1] - state.ci_n_star[0])
    H_width = float(state.ci_H[1] - state.ci_H[0])
    deploy_width_n = params.deploy_width_fraction_n_star * max(
        abs(state.n_star), 1.0e-12
    )
    deploy_width_H = params.deploy_width_fraction_H * max(abs(state.H), 1.0e-12)
    band_width_n = params.band_width_fraction_n_star * max(abs(state.n_star), 1.0e-12)
    band_width_H = params.band_width_fraction_H * max(abs(state.H), 1.0e-12)
    exceed_count = sum(
        state.diagnostics[name] > state.diagnostic_thresholds[name]
        for name in state.diagnostic_thresholds
    )

    if (
        exceed_count
        >= min(params.alarm_min_diagnostics, len(state.diagnostic_thresholds))
        and state.alarm_persistence >= params.persistence_required
    ):
        return ControllerDecision("alarm", "diagnostic exceeded calibrated threshold")

    if (
        n_width <= deploy_width_n
        and H_width <= deploy_width_H
        and state.identifiability_score >= params.S_min
    ):
        return ControllerDecision("use_n_star", "plug-in horizon is precise enough")

    if (
        n_width <= band_width_n
        and H_width <= band_width_H
        and state.identifiability_score >= params.S_min_band
    ):
        return ControllerDecision(
            "use_band", "useful-memory band is more reliable than the point"
        )

    return ControllerDecision("hold", "collect more evidence before deployment")
