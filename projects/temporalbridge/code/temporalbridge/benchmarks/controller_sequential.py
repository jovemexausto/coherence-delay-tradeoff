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
class ControllerSequentialRow:
    policy: str
    repetitions: int
    mean_action_accuracy: float
    mean_false_alarm_rate: float
    mean_over_deployment_rate: float
    mean_under_deployment_rate: float
    mean_cumulative_action_loss: float
    mean_cumulative_validity_loss: float
    mean_cumulative_excess_validity_loss: float
    mean_cumulative_normalized_excess_loss: float
    mean_cumulative_update_cost: float
    mean_update_count: float
    mean_log_memory_std: float
    mean_tau_valid: float
    mean_tau_detect: float
    mean_delay_gap: float
    mean_masking_index: float
    mean_regret: float
    mean_lead_time: float
    median_lead_time: float


@dataclass(frozen=True)
class ControllerSequentialTrajectoryRow:
    trajectory_id: int
    policy: str
    cumulative_action_loss: float
    cumulative_validity_loss: float
    cumulative_excess_validity_loss: float
    cumulative_normalized_excess_loss: float
    cumulative_update_cost: float
    update_count: int
    oracle_loss: float
    regret: float
    lead_time: float
    action_accuracy: float
    false_alarm_rate: float
    log_memory_std: float
    tau_valid: float
    tau_detect: float
    delay_gap: float
    masking_index: float


@dataclass(frozen=True)
class MemoryDynamicsParams:
    tracking_gain: float = 0.35
    band_gain: float = 0.20
    alarm_gain: float = 0.65
    deadband_log: float = 0.08
    update_cost_fixed: float = 0.01
    update_cost_linear: float = 0.15


def _action_loss(expected_action: str, action: str) -> float:
    if action == expected_action:
        return 0.0
    loss_table = {
        ("use_n_star", "use_band"): 0.5,
        ("use_n_star", "hold"): 1.0,
        ("use_n_star", "alarm"): 2.0,
        ("use_band", "use_n_star"): 0.5,
        ("use_band", "hold"): 0.5,
        ("use_band", "alarm"): 1.5,
        ("hold", "use_n_star"): 1.5,
        ("hold", "use_band"): 0.5,
        ("hold", "alarm"): 2.0,
        ("alarm", "hold"): 1.0,
        ("alarm", "use_band"): 1.5,
        ("alarm", "use_n_star"): 2.0,
    }
    return loss_table[(expected_action, action)]


def _schedule_specifications(
    mode: str = "default",
) -> list[tuple[str, int, str]]:
    default_schedule = [
        ("exact", 8, "use_n_star"),
        ("noisy", 6, "hold"),
        ("hetero_power", 6, "use_n_star"),
        ("sinusoidal", 6, "alarm"),
        ("piecewise", 6, "alarm"),
        ("mixed", 6, "alarm"),
        ("strong_sinusoidal", 6, "alarm"),
        ("strong_piecewise", 6, "alarm"),
        ("mixed_strong", 6, "alarm"),
        ("strong_hetero", 6, "alarm"),
        ("exact_recovery", 8, "use_n_star"),
    ]
    if mode == "default":
        return default_schedule
    if mode == "strong":
        return [
            ("strong_sinusoidal", 8, "alarm"),
            ("strong_piecewise", 8, "alarm"),
            ("mixed_strong", 8, "alarm"),
            ("strong_hetero", 8, "alarm"),
        ]
    raise ValueError(f"unsupported schedule mode: {mode}")


def _generate_observation(
    *,
    lags: np.ndarray,
    scenario: str,
    phase: float,
    rng: np.random.Generator,
    truth_h: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    fit_options = {"sigma0": 0.5, "n": 500, "C_K": 20.0, "C_S": 1.0, "a": 0.5}
    if scenario in {"exact", "exact_recovery"}:
        fit_options["C_K"] = 6.0
        true_profile = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=truth_h,
            sigma0=0.0,
            n=500,
            rng=rng,
        )
        return (
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=truth_h,
                sigma0=0.5,
                n=500,
                rng=rng,
            ),
            true_profile,
            fit_options,
        )
    if scenario == "noisy":
        true_profile = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=truth_h,
            sigma0=0.0,
            n=500,
            rng=rng,
        )
        return (
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=truth_h,
                sigma0=1.2,
                n=80,
                rng=rng,
            ),
            true_profile,
            {"sigma0": 1.2, "n": 80, "C_K": 6.0, "C_S": 1.0, "a": 0.5},
        )
    if scenario == "hetero_power":
        alpha = 1.5 + 2.5 * phase
        fit_options["C_K"] = 6.0
        true_profile = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=truth_h,
            sigma0=0.0,
            n=500,
            rng=rng,
        )
        return (
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=truth_h,
                sigma0=0.5,
                n=500,
                noise="heteroskedastic_power",
                heteroskedastic_alpha=alpha,
                heteroskedastic_beta=1.5,
                rng=rng,
            ),
            true_profile,
            fit_options,
        )
    if scenario == "strong_hetero":
        alpha = 3.5 + 3.5 * phase
        fit_options["C_K"] = 2.5
        true_profile = simulate_observed_discrepancies(
            lags,
            zeta=1.0,
            H=truth_h,
            sigma0=0.0,
            n=500,
            rng=rng,
        )
        return (
            simulate_observed_discrepancies(
                lags,
                zeta=1.0,
                H=truth_h,
                sigma0=1.0,
                n=120,
                noise="heteroskedastic_power",
                heteroskedastic_alpha=alpha,
                heteroskedastic_beta=2.5,
                rng=rng,
            ),
            true_profile,
            fit_options,
        )
    if scenario == "sinusoidal":
        fit_options["C_K"] = 3.0
        true_profile = _sinusoidal_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.1 + 0.2 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    if scenario == "strong_sinusoidal":
        fit_options["C_K"] = 2.0
        true_profile = _sinusoidal_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.35 + 0.45 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    if scenario == "piecewise":
        fit_options["C_K"] = 3.0
        true_profile = _piecewise_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.1 + 0.2 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    if scenario == "strong_piecewise":
        fit_options["C_K"] = 2.0
        true_profile = _piecewise_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.35 + 0.45 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    if scenario == "mixed":
        fit_options["C_K"] = 3.0
        true_profile = _mixed_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.1 + 0.2 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    if scenario == "mixed_strong":
        fit_options["C_K"] = 2.0
        true_profile = _mixed_profile(
            lags, zeta=1.0, H=truth_h, amplitude=0.35 + 0.45 * phase
        )
        return (
            true_profile,
            true_profile,
            fit_options,
        )
    raise ValueError(f"unsupported scenario: {scenario}")


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
    *,
    policy: str,
    state: ValidityState,
    expected_action: str,
    params: ControllerParams,
) -> ControllerDecision:
    if policy == "oracle":
        return ControllerDecision(expected_action, "oracle benchmark")
    if policy == "controller":
        return validity_controller(state, params)
    if policy == "fixed_policy":
        return ControllerDecision("hold", "static memory baseline")
    if policy == "deploy_only":
        return ControllerDecision("use_n_star", "always deploy baseline")
    if policy == "detector_only":
        exceed = sum(
            state.diagnostics[name] > state.diagnostic_thresholds[name]
            for name in state.diagnostic_thresholds
        )
        if exceed >= min(
            params.alarm_min_diagnostics, len(state.diagnostic_thresholds)
        ):
            return ControllerDecision("alarm", "detector-only baseline")
        return ControllerDecision("use_n_star", "detector-only baseline")
    raise ValueError(f"unsupported policy: {policy}")


def _choose_memory_level(
    *,
    policy: str,
    action: str,
    state: ValidityState,
    previous_n: float,
    lags: np.ndarray,
    true_profile: np.ndarray,
    dynamics: MemoryDynamicsParams,
) -> float:
    lag_min = float(np.min(lags))
    lag_max = float(np.max(lags))
    if policy == "detector_only" and action != "alarm":
        return previous_n

    if action == "hold":
        target = previous_n
        gain = 0.0
    elif action == "use_n_star":
        target = float(np.clip(state.n_star, lag_min, lag_max))
        gain = dynamics.tracking_gain
    elif action == "use_band":
        midpoint = 0.5 * (state.ci_n_star[0] + state.ci_n_star[1])
        target = float(np.clip(midpoint, lag_min, lag_max))
        gain = dynamics.band_gain
    elif action == "alarm":
        target = float(np.clip(state.ci_n_star[0], lag_min, lag_max))
        gain = dynamics.alarm_gain
    else:
        oracle_idx = int(np.argmin(true_profile))
        target = float(lags[oracle_idx])
        gain = 1.0

    log_previous = float(np.log(max(previous_n, 1.0e-12)))
    log_target = float(np.log(max(target, 1.0e-12)))
    if abs(log_target - log_previous) <= dynamics.deadband_log:
        return previous_n
    log_next = (1.0 - gain) * log_previous + gain * log_target
    return float(np.clip(np.exp(log_next), lag_min, lag_max))


def _profile_loss(true_profile: np.ndarray, lags: np.ndarray, n_value: float) -> float:
    return float(
        np.interp(
            float(n_value),
            np.asarray(lags, dtype=float),
            np.asarray(true_profile, dtype=float),
        )
    )


def _validity_curve(
    *,
    lags: np.ndarray,
    staleness_profile: np.ndarray,
    fit_options: dict[str, float | int],
) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=float)
    C_K = float(fit_options.get("C_K", 1.0))
    C_S = float(fit_options.get("C_S", 1.0))
    a = float(fit_options.get("a", 0.5))
    return C_K * lag_array ** (-a) + C_S * np.asarray(staleness_profile, dtype=float)


def _update_cost(
    previous_n: float, next_n: float, dynamics: MemoryDynamicsParams
) -> float:
    if np.isclose(previous_n, next_n):
        return 0.0
    return dynamics.update_cost_fixed + dynamics.update_cost_linear * abs(
        float(np.log(max(next_n, 1.0e-12)) - np.log(max(previous_n, 1.0e-12)))
    )


def run_controller_sequential_benchmark(
    *,
    repetitions: int = 100,
    bootstrap_method: str = "wild",
    bootstrap_repetitions: int = 20,
    rng_seed: int = 123,
    update_cost_fixed: float = 0.01,
    update_cost_linear: float = 0.15,
    tracking_gain: float = 0.35,
    band_gain: float = 0.20,
    alarm_gain: float = 0.65,
    deadband_log: float = 0.08,
    validity_slack_fraction: float = 0.25,
    schedule_mode: str = "default",
    truth_h: float = 0.6,
) -> dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    lags = np.arange(1, 41, dtype=float)
    thresholds = _calibrated_thresholds(
        lags=lags,
        zeta=1.0,
        H=truth_h,
        sigma0=0.5,
        n=500,
        repetitions=100,
        rng=rng,
    )
    params = ControllerParams()
    dynamics = MemoryDynamicsParams(
        tracking_gain=tracking_gain,
        band_gain=band_gain,
        alarm_gain=alarm_gain,
        deadband_log=deadband_log,
        update_cost_fixed=update_cost_fixed,
        update_cost_linear=update_cost_linear,
    )
    policies = ("oracle", "controller", "fixed_policy", "deploy_only", "detector_only")
    aggregate: dict[str, dict[str, list[float]]] = {
        policy: {
            "accuracy": [],
            "false_alarm_rate": [],
            "over_deployment_rate": [],
            "under_deployment_rate": [],
            "cumulative_action_loss": [],
            "cumulative_validity_loss": [],
            "cumulative_excess_validity_loss": [],
            "cumulative_normalized_excess_loss": [],
            "cumulative_update_cost": [],
            "update_count": [],
            "regret": [],
            "lead_time": [],
            "log_memory_std": [],
            "tau_valid": [],
            "tau_detect": [],
            "delay_gap": [],
            "masking_index": [],
        }
        for policy in policies
    }
    trajectory_rows: list[ControllerSequentialTrajectoryRow] = []

    schedule = _schedule_specifications(mode=schedule_mode)
    for rep in range(repetitions):
        policy_actions = {policy: [] for policy in policies}
        oracle_actions: list[str] = []
        policy_memory: dict[str, float | None] = {policy: None for policy in policies}
        policy_validity_loss: dict[str, float] = {policy: 0.0 for policy in policies}
        policy_excess_validity_loss: dict[str, float] = {
            policy: 0.0 for policy in policies
        }
        policy_normalized_excess_loss: dict[str, float] = {
            policy: 0.0 for policy in policies
        }
        policy_update_cost: dict[str, float] = {policy: 0.0 for policy in policies}
        policy_update_count: dict[str, int] = {policy: 0 for policy in policies}
        policy_memory_trace: dict[str, list[float]] = {
            policy: [] for policy in policies
        }
        policy_step_validity_trace: dict[str, list[float]] = {
            policy: [] for policy in policies
        }
        policy_step_excess_trace: dict[str, list[float]] = {
            policy: [] for policy in policies
        }
        oracle_step_trace: list[float] = []
        oracle_validity_loss = 0.0
        for scenario, duration, expected_action in schedule:
            for step in range(duration):
                phase = 0.0 if duration == 1 else step / float(duration - 1)
                obs, true_profile, fit_options = _generate_observation(
                    lags=lags,
                    scenario=scenario,
                    phase=phase,
                    rng=rng,
                    truth_h=truth_h,
                )
                state = _build_state(
                    lags=lags,
                    obs=obs,
                    fit_options=fit_options,
                    thresholds=thresholds,
                    bootstrap_method=bootstrap_method,
                    bootstrap_repetitions=bootstrap_repetitions,
                    rng_seed=rng_seed + rep,
                )
                oracle_actions.append(expected_action)
                validity_curve = _validity_curve(
                    lags=lags,
                    staleness_profile=true_profile,
                    fit_options=fit_options,
                )
                oracle_n = float(lags[int(np.argmin(validity_curve))])
                oracle_step_loss = _profile_loss(validity_curve, lags, oracle_n)
                oracle_validity_loss += oracle_step_loss
                oracle_step_trace.append(oracle_step_loss)
                for policy in policies:
                    decision = _policy_action(
                        policy=policy,
                        state=state,
                        expected_action=expected_action,
                        params=params,
                    )
                    policy_actions[policy].append(decision.action)
                    prev_n = (
                        float(np.median(lags))
                        if policy_memory[policy] is None
                        else float(policy_memory[policy])
                    )
                    chosen_n = (
                        oracle_n
                        if policy == "oracle"
                        else _choose_memory_level(
                            policy=policy,
                            action=decision.action,
                            state=state,
                            previous_n=prev_n,
                            lags=lags,
                            true_profile=true_profile,
                            dynamics=dynamics,
                        )
                    )
                    policy_update_cost[policy] += _update_cost(
                        prev_n, chosen_n, dynamics
                    )
                    if not np.isclose(prev_n, chosen_n):
                        policy_update_count[policy] += 1
                    policy_memory[policy] = chosen_n
                    policy_memory_trace[policy].append(chosen_n)
                    chosen_loss = _profile_loss(validity_curve, lags, chosen_n)
                    policy_validity_loss[policy] += chosen_loss
                    excess_loss = chosen_loss - oracle_step_loss
                    policy_step_validity_trace[policy].append(chosen_loss)
                    policy_step_excess_trace[policy].append(excess_loss)
                    policy_excess_validity_loss[policy] += excess_loss
                    policy_normalized_excess_loss[policy] += excess_loss / max(
                        oracle_step_loss, 1.0e-12
                    )

        T = len(oracle_actions)
        oracle_cumulative = np.cumsum(np.asarray(oracle_step_trace, dtype=float))
        for policy in policies:
            actions = policy_actions[policy]
            tau_detect = float(
                next((i for i, action in enumerate(actions) if action == "alarm"), T)
            )
            excess_cumulative = np.cumsum(
                np.asarray(policy_step_excess_trace[policy], dtype=float)
            )
            tau_valid = float(
                next(
                    (
                        i
                        for i, excess_value in enumerate(excess_cumulative)
                        if excess_value
                        > validity_slack_fraction
                        * max(float(oracle_cumulative[i]), 1.0e-12)
                    ),
                    T,
                )
            )
            lead_time = float(tau_valid - tau_detect)
            accuracy = float(
                np.mean([a == b for a, b in zip(actions, oracle_actions, strict=False)])
            )
            false_alarm_rate = float(
                np.mean(
                    [
                        actions[i] == "alarm" and oracle_actions[i] != "alarm"
                        for i in range(T)
                    ]
                )
            )
            over_deploy = float(
                np.mean(
                    [
                        actions[i] == "use_n_star" and oracle_actions[i] != "use_n_star"
                        for i in range(T)
                    ]
                )
            )
            under_deploy = float(
                np.mean(
                    [
                        actions[i] != "use_n_star" and oracle_actions[i] == "use_n_star"
                        for i in range(T)
                    ]
                )
            )
            cumulative_loss = float(
                np.sum([_action_loss(oracle_actions[i], actions[i]) for i in range(T)])
            )
            memory_trace = np.asarray(policy_memory_trace[policy], dtype=float)
            log_memory_std = float(np.std(np.log(np.maximum(memory_trace, 1.0e-12))))
            excess_ratio = float(
                policy_excess_validity_loss[policy] / max(oracle_validity_loss, 1.0e-12)
            )
            masking_index = float(accuracy * excess_ratio)
            aggregate[policy]["accuracy"].append(accuracy)
            aggregate[policy]["false_alarm_rate"].append(false_alarm_rate)
            aggregate[policy]["over_deployment_rate"].append(over_deploy)
            aggregate[policy]["under_deployment_rate"].append(under_deploy)
            aggregate[policy]["cumulative_action_loss"].append(cumulative_loss)
            aggregate[policy]["cumulative_validity_loss"].append(
                policy_validity_loss[policy] + policy_update_cost[policy]
            )
            aggregate[policy]["cumulative_excess_validity_loss"].append(
                policy_excess_validity_loss[policy]
            )
            aggregate[policy]["cumulative_normalized_excess_loss"].append(
                policy_normalized_excess_loss[policy]
            )
            aggregate[policy]["cumulative_update_cost"].append(
                policy_update_cost[policy]
            )
            aggregate[policy]["update_count"].append(float(policy_update_count[policy]))
            aggregate[policy]["log_memory_std"].append(log_memory_std)
            aggregate[policy]["tau_valid"].append(tau_valid)
            aggregate[policy]["tau_detect"].append(tau_detect)
            aggregate[policy]["delay_gap"].append(float(tau_detect - tau_valid))
            aggregate[policy]["masking_index"].append(masking_index)
            aggregate[policy]["regret"].append(
                (policy_validity_loss[policy] + policy_update_cost[policy])
                - (oracle_validity_loss + policy_update_cost["oracle"])
            )
            aggregate[policy]["lead_time"].append(lead_time)
            trajectory_rows.append(
                ControllerSequentialTrajectoryRow(
                    trajectory_id=rep,
                    policy=policy,
                    cumulative_action_loss=cumulative_loss,
                    cumulative_validity_loss=policy_validity_loss[policy]
                    + policy_update_cost[policy],
                    cumulative_excess_validity_loss=policy_excess_validity_loss[policy],
                    cumulative_normalized_excess_loss=policy_normalized_excess_loss[
                        policy
                    ],
                    cumulative_update_cost=policy_update_cost[policy],
                    update_count=policy_update_count[policy],
                    oracle_loss=oracle_validity_loss + policy_update_cost["oracle"],
                    regret=(policy_validity_loss[policy] + policy_update_cost[policy])
                    - (oracle_validity_loss + policy_update_cost["oracle"]),
                    lead_time=lead_time,
                    action_accuracy=accuracy,
                    false_alarm_rate=false_alarm_rate,
                    log_memory_std=log_memory_std,
                    tau_valid=tau_valid,
                    tau_detect=tau_detect,
                    delay_gap=float(tau_detect - tau_valid),
                    masking_index=masking_index,
                )
            )

    rows = [
        ControllerSequentialRow(
            policy=policy,
            repetitions=repetitions,
            mean_action_accuracy=float(np.mean(stats["accuracy"])),
            mean_false_alarm_rate=float(np.mean(stats["false_alarm_rate"])),
            mean_over_deployment_rate=float(np.mean(stats["over_deployment_rate"])),
            mean_under_deployment_rate=float(np.mean(stats["under_deployment_rate"])),
            mean_cumulative_action_loss=float(np.mean(stats["cumulative_action_loss"])),
            mean_cumulative_validity_loss=float(
                np.mean(stats["cumulative_validity_loss"])
            ),
            mean_cumulative_excess_validity_loss=float(
                np.mean(stats["cumulative_excess_validity_loss"])
            ),
            mean_cumulative_normalized_excess_loss=float(
                np.mean(stats["cumulative_normalized_excess_loss"])
            ),
            mean_cumulative_update_cost=float(np.mean(stats["cumulative_update_cost"])),
            mean_update_count=float(np.mean(stats["update_count"])),
            mean_log_memory_std=float(np.mean(stats["log_memory_std"])),
            mean_tau_valid=float(np.mean(stats["tau_valid"])),
            mean_tau_detect=float(np.mean(stats["tau_detect"])),
            mean_delay_gap=float(np.mean(stats["delay_gap"])),
            mean_masking_index=float(np.mean(stats["masking_index"])),
            mean_regret=float(np.mean(stats["regret"])),
            mean_lead_time=float(np.mean(stats["lead_time"])),
            median_lead_time=float(np.median(stats["lead_time"])),
        )
        for policy, stats in aggregate.items()
    ]
    return {
        "rows": [asdict(row) for row in rows],
        "trajectory_rows": [asdict(row) for row in trajectory_rows],
        "thresholds": thresholds,
        "bootstrap_method": bootstrap_method,
    }
