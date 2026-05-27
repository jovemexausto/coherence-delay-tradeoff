from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .partition_ratio import ratio_at_optimizer, validity_partition_ratio
from .ratio_control import RatioControlConfig, simulate_ratio_tracking


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_temporalbridge_path() -> None:
    code_root = _project_root() / "projects" / "temporalbridge" / "code"
    if code_root.exists() and str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))


def _ensure_scale_consistency_path() -> None:
    code_root = _project_root() / "projects" / "scale-consistency" / "code"
    if code_root.exists() and str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))


@dataclass(frozen=True, slots=True)
class RegimeProxy:
    H_hat: float
    zeta_hat: float
    noise_hat: float
    identifiability: float


@dataclass(frozen=True, slots=True)
class PolicyRouterConfig:
    H_values: tuple[float, ...] = (0.3, 0.5, 0.75, 0.9)
    zeta0_values: tuple[float, ...] = (0.01, 0.05)
    ramp_values: tuple[float, ...] = (2.0e-5, 5.0e-5)
    noise_values: tuple[float, ...] = (0.05, 0.15, 0.3)
    steps: int = 128
    lag_count: int = 40
    lag_reps: int = 12
    sigma0: float = 0.5
    n_obs: int = 500
    a: float = 0.5
    C_K: float = 1.0
    C_S: float = 1.0
    n0: float = 16.0
    n_min: float = 2.0
    n_max: float = 4096.0
    instant_noise_threshold: float = 0.22
    lag_noise_threshold: float = 0.18
    high_H_threshold: float = 0.48
    identifiability_threshold: float = 0.0
    switch_persistence: int = 4
    persistent_decay: float = 0.85
    persistent_gain: float = 0.75
    persistent_deadband: float = 0.10


@dataclass(frozen=True, slots=True)
class PolicyRouterRow:
    policy: str
    H: float
    zeta0: float
    ramp: float
    noise: float
    mean_relative_error: float
    std_relative_error: float
    mean_abs_log_update: float
    final_relative_error: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _estimate_proxy(
    *,
    lags: np.ndarray,
    discrepancy: np.ndarray,
    rho_window: np.ndarray,
    config: PolicyRouterConfig,
) -> RegimeProxy:
    _ensure_temporalbridge_path()
    _ensure_scale_consistency_path()
    from temporalbridge.core.fit import fit_horizon

    fit = fit_horizon(
        lags,
        discrepancy,
        fit_options={
            "sigma0": config.sigma0,
            "n": config.n_obs,
            "C_K": config.C_K,
            "C_S": config.C_S,
            "a": config.a,
        },
    )
    residuals = np.asarray(fit["fit_stats"]["residuals"], dtype=float)
    rho_window = np.asarray(rho_window, dtype=float)
    rho_window = rho_window[np.isfinite(rho_window) & (rho_window > 0.0)]
    if rho_window.size >= 2:
        noise_hat = float(np.std(np.log(rho_window)))
    else:
        noise_hat = float(np.std(residuals))
    return RegimeProxy(
        H_hat=float(max(fit["H"], 0.05)),
        zeta_hat=float(fit["zeta"]),
        noise_hat=noise_hat,
        identifiability=float(1.0 / (1.0 + float(fit["fit_stats"]["loss"]))),
    )


def _lag_geometry_proposal(
    *, config: PolicyRouterConfig, lags: np.ndarray, discrepancy: np.ndarray
) -> float:
    _ensure_temporalbridge_path()
    from temporalbridge.core.fit import fit_horizon

    fit = fit_horizon(
        lags,
        discrepancy,
        fit_options={
            "sigma0": config.sigma0,
            "n": config.n_obs,
            "C_K": config.C_K,
            "C_S": config.C_S,
            "a": config.a,
        },
    )
    return _clamp(float(fit["n_star"]), config.n_min, config.n_max)


def _instant_ratio_proposal(
    *, config: PolicyRouterConfig, current_n: float, rho_obs: float, H_hat: float
) -> float:
    target = ratio_at_optimizer(config.a, H_hat)
    return _clamp(
        current_n * ((target / max(rho_obs, 1.0e-12)) ** (1.0 / (config.a + H_hat))),
        config.n_min,
        config.n_max,
    )


def _persistent_ratio_proposal(
    *,
    config: PolicyRouterConfig,
    current_n: float,
    rho_obs: float,
    H_hat: float,
    score: float,
) -> tuple[float, float]:
    target = ratio_at_optimizer(config.a, H_hat)
    evidence = float(np.log(max(rho_obs, 1.0e-12) / target))
    score = config.persistent_decay * score + (1.0 - config.persistent_decay) * evidence
    if abs(score) <= config.persistent_deadband:
        return current_n, score
    next_n = _clamp(
        current_n * np.exp(-config.persistent_gain * score), config.n_min, config.n_max
    )
    return next_n, score


def _route_policy(proxy: RegimeProxy, config: PolicyRouterConfig) -> str:
    if proxy.noise_hat >= config.instant_noise_threshold:
        return "persistent_ratio"
    if (
        proxy.H_hat >= config.high_H_threshold
        and proxy.noise_hat <= config.lag_noise_threshold
    ):
        return "lag_geometry"
    return "instant_ratio"


def run_policy_router_benchmark(
    *,
    config: PolicyRouterConfig = PolicyRouterConfig(),
    rng_seed: int = 0,
) -> list[PolicyRouterRow]:
    _ensure_temporalbridge_path()
    _ensure_scale_consistency_path()
    from scale_consistency.model import simulate_observed_discrepancies

    rows: list[PolicyRouterRow] = []
    rng = np.random.default_rng(rng_seed)
    lags = np.arange(1, config.lag_count + 1, dtype=float)

    for H in config.H_values:
        for zeta0 in config.zeta0_values:
            for ramp in config.ramp_values:
                for noise in config.noise_values:
                    ratio_config = RatioControlConfig(
                        a=config.a,
                        H=H,
                        C_K=config.C_K,
                        C_S=config.C_S,
                        zeta0=zeta0,
                        ramp=ramp,
                        steps=config.steps,
                        noise_sigma=noise,
                        n0=config.n0,
                        n_min=config.n_min,
                        n_max=config.n_max,
                        decay=config.persistent_decay,
                        gain=config.persistent_gain,
                        deadband=config.persistent_deadband,
                    )
                    ratio_path = simulate_ratio_tracking(
                        ratio_config, policy="instant", rng_seed=rng_seed
                    )
                    policies = {
                        "instant_ratio": {"n": float(config.n0), "err": [], "upd": []},
                        "persistent_ratio": {
                            "n": float(config.n0),
                            "err": [],
                            "upd": [],
                            "score": 0.0,
                        },
                        "lag_geometry": {"n": float(config.n0), "err": [], "upd": []},
                        "regime_router": {
                            "n": float(config.n0),
                            "err": [],
                            "upd": [],
                            "score": 0.0,
                            "current_policy": "persistent_ratio",
                            "streak": 0,
                        },
                    }

                    for t in range(config.steps):
                        zeta = zeta0 + ramp * t
                        n_star = (
                            (config.a * config.C_K) / (H * config.C_S * zeta)
                        ) ** (1.0 / (config.a + H))
                        discrepancy = np.mean(
                            [
                                simulate_observed_discrepancies(
                                    lags,
                                    zeta=float(zeta),
                                    H=H,
                                    sigma0=config.sigma0,
                                    n=config.n_obs,
                                    noise="heteroskedastic_power"
                                    if noise > 0.12
                                    else "gaussian",
                                    heteroskedastic_alpha=noise,
                                    heteroskedastic_beta=1.5,
                                    rng=np.random.default_rng(
                                        int(rng.integers(0, 2**32 - 1))
                                    ),
                                )
                                for _ in range(config.lag_reps)
                            ],
                            axis=0,
                        )
                        rho_obs = float(ratio_path.rho_obs_path[t])
                        rho_window = ratio_path.rho_obs_path[
                            max(0, t - config.lag_count + 1) : t + 1
                        ]
                        proxy = _estimate_proxy(
                            lags=lags,
                            discrepancy=discrepancy,
                            rho_window=rho_window,
                            config=config,
                        )

                        n_inst = _instant_ratio_proposal(
                            config=config,
                            current_n=policies["instant_ratio"]["n"],
                            rho_obs=rho_obs,
                            H_hat=H,
                        )
                        n_persist, new_score = _persistent_ratio_proposal(
                            config=config,
                            current_n=policies["persistent_ratio"]["n"],
                            rho_obs=rho_obs,
                            H_hat=H,
                            score=float(policies["persistent_ratio"]["score"]),
                        )
                        n_lag = _lag_geometry_proposal(
                            config=config, lags=lags, discrepancy=discrepancy
                        )

                        chosen = _route_policy(proxy, config)
                        router_state = policies["regime_router"]
                        if chosen == router_state["current_policy"]:
                            router_state["streak"] = 0
                        else:
                            router_state["streak"] = int(router_state["streak"]) + 1
                            if router_state["streak"] >= config.switch_persistence:
                                router_state["current_policy"] = chosen
                                router_state["streak"] = 0
                        chosen_router = str(router_state["current_policy"])
                        n_router = {
                            "instant_ratio": n_inst,
                            "persistent_ratio": n_persist,
                            "lag_geometry": n_lag,
                        }[chosen_router]

                        candidates = {
                            "instant_ratio": n_inst,
                            "persistent_ratio": n_persist,
                            "lag_geometry": n_lag,
                            "regime_router": n_router,
                        }
                        for policy, state in policies.items():
                            n_prev = float(state["n"])
                            n_next = (
                                candidates[policy]
                                if policy != "regime_router"
                                else n_router
                            )
                            state["err"].append(
                                abs(n_prev - n_star) / max(n_star, 1.0e-12)
                            )
                            state["upd"].append(
                                abs(np.log(max(n_next, 1.0e-12) / max(n_prev, 1.0e-12)))
                            )
                            state["n"] = n_next
                        policies["persistent_ratio"]["score"] = new_score
                        policies["regime_router"]["score"] = (
                            new_score
                            if chosen_router == "persistent_ratio"
                            else float(policies["regime_router"]["score"])
                        )

                    for policy, state in policies.items():
                        errors = np.asarray(state["err"], dtype=float)
                        updates = np.asarray(state["upd"], dtype=float)
                        rows.append(
                            PolicyRouterRow(
                                policy=policy,
                                H=H,
                                zeta0=zeta0,
                                ramp=ramp,
                                noise=noise,
                                mean_relative_error=float(np.mean(errors)),
                                std_relative_error=float(np.std(errors)),
                                mean_abs_log_update=float(np.mean(updates)),
                                final_relative_error=float(errors[-1]),
                            )
                        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy-router benchmark.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = PolicyRouterConfig(steps=args.steps, noise_values=(args.noise,))
    rows = run_policy_router_benchmark(config=config, rng_seed=0)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print(
        "policy\tH\tzeta0\tramp\tnoise\tmean_rel_err\tmean_abs_log_update\tfinal_rel_err"
    )
    for row in rows:
        print(
            f"{row.policy}\t{row.H:.2f}\t{row.zeta0:.3f}\t{row.ramp:.2e}\t{row.noise:.3f}\t"
            f"{row.mean_relative_error:.4f}\t{row.mean_abs_log_update:.4f}\t{row.final_relative_error:.4f}"
        )


if __name__ == "__main__":
    main()
