from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from temporalbridge._backend import simulate_observed_discrepancies
from temporalbridge.core.alarms import calibrate_alarms
from temporalbridge.core.bootstrap import bootstrap_horizon
from temporalbridge.core.controller import (
    ControllerParams,
    ValidityState,
    validity_controller,
)
from temporalbridge.core.fit import fit_horizon
from temporalbridge.utils.diagnostics import compute_profile_diagnostics


@dataclass(frozen=True)
class ControllerBenchmarkRow:
    scenario: str
    expected_action: str
    action: str
    H: float
    n_star: float
    ci_width_n_star: float
    ci_width_H: float
    identifiability_score: float
    KL_residual: float
    KL_standardized: float


def _exact_profile(lags: np.ndarray, *, zeta: float, H: float) -> np.ndarray:
    return zeta * np.asarray(lags, dtype=float) ** H


def _sinusoidal_profile(
    lags: np.ndarray, *, zeta: float, H: float, amplitude: float
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    x = np.log(lag_array)
    span = max(float(np.max(x) - np.min(x)), 1.0e-8)
    return _exact_profile(lag_array, zeta=zeta, H=H) * (
        1.0 + amplitude * np.sin(2.0 * np.pi * (x - np.min(x)) / span)
    )


def _piecewise_profile(
    lags: np.ndarray, *, zeta: float, H: float, amplitude: float
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    midpoint = float(np.median(lag_array))
    left = lag_array <= midpoint
    profile = np.empty_like(lag_array)
    profile[left] = zeta * lag_array[left] ** H
    continuity_scale = midpoint ** (-amplitude)
    profile[~left] = zeta * continuity_scale * lag_array[~left] ** (H + amplitude)
    return profile


def _mixed_profile(
    lags: np.ndarray, *, zeta: float, H: float, amplitude: float
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    base = _exact_profile(lag_array, zeta=zeta, H=H)
    oscillatory = _sinusoidal_profile(
        lag_array, zeta=1.0, H=0.0, amplitude=0.5 * amplitude
    )
    jumps = np.exp(amplitude * (lag_array >= np.quantile(lag_array, 0.7)))
    return base * oscillatory * jumps


def _summarize_diagnostics(
    diagnostics: dict[str, np.ndarray | float],
) -> dict[str, float]:
    return {
        "KL_residual": float(
            np.max(np.asarray(diagnostics["KL_residual"], dtype=float))
        ),
        "KL_standardized": float(
            np.max(np.asarray(diagnostics["KL_standardized"], dtype=float))
        ),
    }


def _compute_identifiability(
    profile: dict[str, object], bootstrap: dict[str, object]
) -> float:
    n_star = float(profile["n_star"])
    ci_n_star = bootstrap["ci_n_star"]
    width = float(ci_n_star[1] - ci_n_star[0])
    return float(n_star / max(n_star + width, 1.0e-12))


def _calibrated_thresholds(
    *,
    lags: np.ndarray,
    zeta: float,
    H: float,
    sigma0: float,
    n: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    null_summary = {"KL_residual": [], "KL_standardized": []}
    for _ in range(repetitions):
        obs = simulate_observed_discrepancies(
            lags,
            zeta=zeta,
            H=H,
            sigma0=sigma0,
            n=n,
            rng=rng,
        )
        profile = fit_horizon(lags, obs, fit_options={"sigma0": sigma0, "n": n})
        summary = _summarize_diagnostics(compute_profile_diagnostics(profile))
        for key, value in summary.items():
            null_summary[key].append(value)
    calibrated = calibrate_alarms(
        profile={},
        bootstrap_results={"method": "wild", "diagnostic_bootstrap": null_summary},
        diagnostics=null_summary,
    )
    return {
        name: float(levels["q95"]) for name, levels in calibrated["thresholds"].items()
    }


def run_controller_benchmark(
    *,
    rng_seed: int = 123,
    bootstrap_method: str = "wild",
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
    scenarios = []

    exact_obs = simulate_observed_discrepancies(
        lags,
        zeta=1.0,
        H=0.6,
        sigma0=0.5,
        n=500,
        rng=rng,
    )
    scenarios.append(("exact", "use_n_star", exact_obs, {"sigma0": 0.5, "n": 500}))

    noisy_obs = simulate_observed_discrepancies(
        lags,
        zeta=1.0,
        H=0.6,
        sigma0=1.2,
        n=80,
        rng=rng,
    )
    scenarios.append(("noisy", "hold", noisy_obs, {"sigma0": 1.2, "n": 80}))

    sinusoidal_obs = _sinusoidal_profile(lags, zeta=1.0, H=0.6, amplitude=0.3)
    scenarios.append(
        ("sinusoidal_misspec", "alarm", sinusoidal_obs, {"sigma0": 0.5, "n": 500})
    )

    rows: list[ControllerBenchmarkRow] = []
    for scenario, expected_action, obs, fit_options in scenarios:
        profile = fit_horizon(lags, obs, fit_options=fit_options)
        bootstrap = bootstrap_horizon(
            profile,
            method=bootstrap_method,
            n_boot=200,
            rng_seed=rng_seed,
        )
        summary = _summarize_diagnostics(compute_profile_diagnostics(profile))
        state = ValidityState(
            n_star=float(profile["n_star"]),
            ci_n_star=bootstrap["ci_n_star"],
            H=float(profile["H"]),
            ci_H=bootstrap["ci_H"],
            identifiability_score=_compute_identifiability(profile, bootstrap),
            diagnostics=summary,
            diagnostic_thresholds=thresholds,
            alarm_persistence=1,
            bootstrap_mode=bootstrap_method,
        )
        decision = validity_controller(state, params)
        rows.append(
            ControllerBenchmarkRow(
                scenario=scenario,
                expected_action=expected_action,
                action=decision.action,
                H=state.H,
                n_star=state.n_star,
                ci_width_n_star=float(state.ci_n_star[1] - state.ci_n_star[0]),
                ci_width_H=float(state.ci_H[1] - state.ci_H[0]),
                identifiability_score=state.identifiability_score,
                KL_residual=summary["KL_residual"],
                KL_standardized=summary["KL_standardized"],
            )
        )

    accuracy = float(np.mean([row.action == row.expected_action for row in rows]))
    return {
        "rows": [asdict(row) for row in rows],
        "accuracy": accuracy,
        "thresholds": thresholds,
        "bootstrap_method": bootstrap_method,
    }


def run_controller_grid_benchmark(
    *,
    rng_seed: int = 123,
    bootstrap_method: str = "wild",
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
    scenarios = [
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

    rows: list[ControllerBenchmarkRow] = []
    for scenario, expected_action, obs, fit_options in scenarios:
        profile = fit_horizon(lags, obs, fit_options=fit_options)
        bootstrap = bootstrap_horizon(
            profile,
            method=bootstrap_method,
            n_boot=200,
            rng_seed=rng_seed,
        )
        summary = _summarize_diagnostics(compute_profile_diagnostics(profile))
        state = ValidityState(
            n_star=float(profile["n_star"]),
            ci_n_star=bootstrap["ci_n_star"],
            H=float(profile["H"]),
            ci_H=bootstrap["ci_H"],
            identifiability_score=_compute_identifiability(profile, bootstrap),
            diagnostics=summary,
            diagnostic_thresholds=thresholds,
            alarm_persistence=1,
            bootstrap_mode=bootstrap_method,
        )
        decision = validity_controller(state, params)
        rows.append(
            ControllerBenchmarkRow(
                scenario=scenario,
                expected_action=expected_action,
                action=decision.action,
                H=state.H,
                n_star=state.n_star,
                ci_width_n_star=float(state.ci_n_star[1] - state.ci_n_star[0]),
                ci_width_H=float(state.ci_H[1] - state.ci_H[0]),
                identifiability_score=state.identifiability_score,
                KL_residual=summary["KL_residual"],
                KL_standardized=summary["KL_standardized"],
            )
        )

    accuracy = float(np.mean([row.action == row.expected_action for row in rows]))
    return {
        "rows": [asdict(row) for row in rows],
        "accuracy": accuracy,
        "thresholds": thresholds,
        "bootstrap_method": bootstrap_method,
    }
