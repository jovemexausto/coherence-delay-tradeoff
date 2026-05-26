from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from temporalbridge._backend import simulate_observed_discrepancies
from temporalbridge.benchmarks.controller_benchmark import (
    _calibrated_thresholds,
    _compute_identifiability,
    _mixed_profile,
    _piecewise_profile,
    _sinusoidal_profile,
    _summarize_diagnostics,
)
from temporalbridge.core.bootstrap import bootstrap_horizon
from temporalbridge.core.controller import (
    ControllerDecision,
    ControllerParams,
    ValidityState,
    validity_controller,
)
from temporalbridge.core.fit import fit_horizon
from temporalbridge.utils.diagnostics import compute_profile_diagnostics


@dataclass(frozen=True)
class ControllerMonteCarloRow:
    scenario: str
    policy: str
    repetitions: int
    accuracy: float
    false_alarm_rate: float
    miss_alarm_rate: float
    over_deployment_rate: float
    under_deployment_rate: float
    mean_action_loss: float


def _build_state(
    *,
    lags: np.ndarray,
    obs: np.ndarray,
    fit_options: dict[str, float | int],
    thresholds: dict[str, float],
    bootstrap_method: str,
    bootstrap_repetitions: int,
    rng_seed: int,
) -> ValidityState:
    profile = fit_horizon(lags, obs, fit_options=fit_options)
    bootstrap = bootstrap_horizon(
        profile,
        method=bootstrap_method,
        n_boot=bootstrap_repetitions,
        rng_seed=rng_seed,
    )
    diagnostics = _summarize_diagnostics(compute_profile_diagnostics(profile))
    return ValidityState(
        n_star=float(profile["n_star"]),
        ci_n_star=bootstrap["ci_n_star"],
        H=float(profile["H"]),
        ci_H=bootstrap["ci_H"],
        identifiability_score=_compute_identifiability(profile, bootstrap),
        diagnostics=diagnostics,
        diagnostic_thresholds=thresholds,
        alarm_persistence=1,
        bootstrap_mode=bootstrap_method,
    )


def _policy_action(
    state: ValidityState,
    *,
    expected_action: str,
    params: ControllerParams,
    policy: str,
) -> ControllerDecision:
    if policy == "oracle":
        return ControllerDecision(expected_action, "oracle benchmark")
    if policy == "fixed_policy":
        return ControllerDecision("use_n_star", "static deploy baseline")
    if policy == "detector_only":
        exceed_count = sum(
            state.diagnostics[name] > state.diagnostic_thresholds[name]
            for name in state.diagnostic_thresholds
        )
        if exceed_count >= min(
            params.alarm_min_diagnostics, len(state.diagnostic_thresholds)
        ):
            return ControllerDecision("alarm", "detector-only baseline")
        return ControllerDecision("use_n_star", "detector-only baseline")
    if policy == "controller":
        return validity_controller(state, params)
    raise ValueError(f"unsupported policy: {policy}")


def _action_loss(expected_action: str, action: str) -> float:
    if action == expected_action:
        return 0.0
    loss_table = {
        ("use_n_star", "use_band"): 0.5,
        ("use_n_star", "hold"): 1.0,
        ("use_n_star", "alarm"): 2.0,
        ("hold", "use_n_star"): 1.5,
        ("hold", "use_band"): 0.5,
        ("hold", "alarm"): 2.0,
        ("alarm", "hold"): 1.0,
        ("alarm", "use_band"): 1.5,
        ("alarm", "use_n_star"): 2.0,
    }
    return loss_table[(expected_action, action)]


def _scenario_specifications(
    rng: np.random.Generator,
) -> list[tuple[str, str, np.ndarray, dict[str, float | int]]]:
    lags = np.arange(1, 41, dtype=float)
    return [
        (
            "exact",
            "use_n_star",
            simulate_observed_discrepancies(
                lags, zeta=1.0, H=0.6, sigma0=0.5, n=500, rng=rng
            ),
            {"sigma0": 0.5, "n": 500},
        ),
        (
            "noisy",
            "hold",
            simulate_observed_discrepancies(
                lags, zeta=1.0, H=0.6, sigma0=1.2, n=80, rng=rng
            ),
            {"sigma0": 1.2, "n": 80},
        ),
        (
            "sinusoidal_misspec",
            "alarm",
            _sinusoidal_profile(lags, zeta=1.0, H=0.6, amplitude=0.3),
            {"sigma0": 0.5, "n": 500},
        ),
        (
            "piecewise_misspec",
            "alarm",
            _piecewise_profile(lags, zeta=1.0, H=0.6, amplitude=0.3),
            {"sigma0": 0.5, "n": 500},
        ),
        (
            "mixed_misspec",
            "alarm",
            _mixed_profile(lags, zeta=1.0, H=0.6, amplitude=0.2),
            {"sigma0": 0.5, "n": 500},
        ),
        (
            "hetero_power",
            "use_n_star",
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=0.6,
                sigma0=0.5,
                n=500,
                noise="heteroskedastic_power",
                heteroskedastic_alpha=4.0,
                heteroskedastic_beta=1.5,
                rng=rng,
            ),
            {"sigma0": 0.5, "n": 500},
        ),
        (
            "hetero_ar",
            "use_n_star",
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=0.6,
                sigma0=0.5,
                n=500,
                noise="heteroskedastic_ar",
                heteroskedastic_alpha=0.35,
                heteroskedastic_rho=0.8,
                rng=rng,
            ),
            {"sigma0": 0.5, "n": 500},
        ),
    ]


def run_controller_monte_carlo(
    *,
    repetitions: int = 50,
    bootstrap_method: str = "wild",
    bootstrap_repetitions: int = 100,
    rng_seed: int = 123,
) -> dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    lags = np.arange(1, 41, dtype=float)
    thresholds = _calibrated_thresholds(
        lags=lags,
        zeta=1.0,
        H=0.6,
        sigma0=0.5,
        n=500,
        repetitions=100,
        rng=rng,
    )
    params = ControllerParams()
    policies = ("oracle", "controller", "fixed_policy", "detector_only")
    stats: dict[tuple[str, str], dict[str, list[float]]] = {}

    for rep in range(repetitions):
        for scenario, expected_action, obs, fit_options in _scenario_specifications(
            rng
        ):
            state = _build_state(
                lags=lags,
                obs=obs,
                fit_options=fit_options,
                thresholds=thresholds,
                bootstrap_method=bootstrap_method,
                bootstrap_repetitions=bootstrap_repetitions,
                rng_seed=rng_seed + rep,
            )
            for policy in policies:
                decision = _policy_action(
                    state,
                    expected_action=expected_action,
                    params=params,
                    policy=policy,
                )
                key = (scenario, policy)
                bucket = stats.setdefault(
                    key,
                    {
                        "correct": [],
                        "false_alarm": [],
                        "miss_alarm": [],
                        "over_deploy": [],
                        "under_deploy": [],
                        "loss": [],
                    },
                )
                bucket["correct"].append(float(decision.action == expected_action))
                bucket["false_alarm"].append(
                    float(expected_action != "alarm" and decision.action == "alarm")
                )
                bucket["miss_alarm"].append(
                    float(expected_action == "alarm" and decision.action != "alarm")
                )
                bucket["over_deploy"].append(
                    float(
                        expected_action != "use_n_star"
                        and decision.action == "use_n_star"
                    )
                )
                bucket["under_deploy"].append(
                    float(
                        expected_action == "use_n_star"
                        and decision.action != "use_n_star"
                    )
                )
                bucket["loss"].append(_action_loss(expected_action, decision.action))

    rows = [
        ControllerMonteCarloRow(
            scenario=scenario,
            policy=policy,
            repetitions=repetitions,
            accuracy=float(np.mean(bucket["correct"])),
            false_alarm_rate=float(np.mean(bucket["false_alarm"])),
            miss_alarm_rate=float(np.mean(bucket["miss_alarm"])),
            over_deployment_rate=float(np.mean(bucket["over_deploy"])),
            under_deployment_rate=float(np.mean(bucket["under_deploy"])),
            mean_action_loss=float(np.mean(bucket["loss"])),
        )
        for (scenario, policy), bucket in sorted(stats.items())
    ]
    aggregated = []
    for policy in policies:
        subset = [row for row in rows if row.policy == policy]
        aggregated.append(
            {
                "policy": policy,
                "accuracy": float(np.mean([row.accuracy for row in subset])),
                "false_alarm_rate": float(
                    np.mean([row.false_alarm_rate for row in subset])
                ),
                "miss_alarm_rate": float(
                    np.mean([row.miss_alarm_rate for row in subset])
                ),
                "over_deployment_rate": float(
                    np.mean([row.over_deployment_rate for row in subset])
                ),
                "under_deployment_rate": float(
                    np.mean([row.under_deployment_rate for row in subset])
                ),
                "mean_action_loss": float(
                    np.mean([row.mean_action_loss for row in subset])
                ),
            }
        )
    return {
        "rows": [asdict(row) for row in rows],
        "aggregated": aggregated,
        "bootstrap_method": bootstrap_method,
        "thresholds": thresholds,
    }
