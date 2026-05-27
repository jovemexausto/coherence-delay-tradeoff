from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json

import numpy as np

from .policy_router import (
    PolicyRouterConfig,
    RegimeProxy,
    _estimate_proxy,
    _route_policy,
)
from .regime_route_delay import (
    RegimeCell,
    _mismatch_cost,
    _route_with_persistence,
    _clamp,
)
from .ratio_control import RatioControlConfig, simulate_ratio_tracking


@dataclass(frozen=True, slots=True)
class MetaSensingConfig:
    pre_regime: RegimeCell = RegimeCell(0.3, 0.01, 0.05, "instant_ratio")
    post_regime: RegimeCell = RegimeCell(0.9, 0.01, 0.05, "lag_geometry")
    steps: int = 128
    switch_step: int = 64
    trials: int = 64
    lag_count: int = 40
    lag_reps: int = 12
    noise_sigma: float = 0.5
    n_obs: int = 500
    sensor_noise_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    switch_persistence: int = 4
    instant_noise_threshold: float = 0.22
    lag_noise_threshold: float = 0.18
    high_H_threshold: float = 0.48
    H_shift_threshold: float = 0.12
    noise_shift_threshold: float = 0.08


@dataclass(frozen=True, slots=True)
class MetaSensingRow:
    sensor_mode: str
    sensor_noise: float
    mean_route_delay: float
    median_route_delay: float
    mean_pre_route_cost: float
    mean_switch_accuracy: float
    mean_route_lag_to_oracle: float


def _sensor_proxy(
    cell: RegimeCell, rng: np.random.Generator, sigma: float
) -> RegimeProxy:
    noise_hat = max(cell.noise + rng.normal(0.0, sigma), 0.0)
    return RegimeProxy(
        H_hat=max(cell.H + rng.normal(0.0, sigma), 0.05),
        zeta_hat=max(cell.zeta * np.exp(rng.normal(0.0, sigma)), 1.0e-12),
        noise_hat=noise_hat,
        identifiability=float(1.0 / (1.0 + noise_hat)),
    )


def _route_meta_sensing(
    *,
    full_proxy: RegimeProxy,
    short_proxy: RegimeProxy,
    long_proxy: RegimeProxy,
    config: MetaSensingConfig,
) -> str:
    class _Cfg:
        instant_noise_threshold = config.instant_noise_threshold
        lag_noise_threshold = config.lag_noise_threshold
        high_H_threshold = config.high_H_threshold

    votes = [
        _route_policy(full_proxy, _Cfg()),
        _route_policy(short_proxy, _Cfg()),
        _route_policy(long_proxy, _Cfg()),
    ]
    counts = {policy: votes.count(policy) for policy in set(votes)}
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    avg_noise = (
        full_proxy.noise_hat + short_proxy.noise_hat + long_proxy.noise_hat
    ) / 3.0
    avg_h = (full_proxy.H_hat + short_proxy.H_hat + long_proxy.H_hat) / 3.0
    if avg_noise >= config.instant_noise_threshold:
        return "persistent_ratio"
    if avg_h >= config.high_H_threshold and avg_noise <= config.lag_noise_threshold:
        return "lag_geometry"
    return "instant_ratio"


def run_meta_sensing_benchmark(
    *,
    config: MetaSensingConfig = MetaSensingConfig(),
    rng_seed: int = 0,
) -> list[MetaSensingRow]:
    rows: list[MetaSensingRow] = []
    rng = np.random.default_rng(rng_seed)
    lags = np.arange(1, config.lag_count + 1, dtype=float)
    fit_config = PolicyRouterConfig(
        lag_count=config.lag_count,
        lag_reps=config.lag_reps,
        sigma0=config.noise_sigma,
        n_obs=config.n_obs,
        a=0.5,
        C_K=1.0,
        C_S=1.0,
    )

    for sensor_noise in config.sensor_noise_levels:
        for sensor_mode in ("single", "multiscale"):
            route_delays: list[float] = []
            pre_route_costs: list[float] = []
            switch_accuracies: list[float] = []
            route_lag_to_oracle: list[float] = []

            for _trial in range(config.trials):
                current_policy = config.pre_regime.oracle_policy
                streak = 0
                route_time = config.steps - 1
                oracle_switch_time = config.switch_step
                correct_after_switch = 0
                total_after_switch = 0
                pre_route_cost = 0.0

                for t in range(config.steps):
                    latent = (
                        config.pre_regime
                        if t < config.switch_step
                        else config.post_regime
                    )
                    true_proxy = RegimeProxy(
                        H_hat=latent.H,
                        zeta_hat=latent.zeta,
                        noise_hat=latent.noise,
                        identifiability=float(1.0 / (1.0 + latent.noise)),
                    )
                    observed_proxy = _sensor_proxy(latent, rng, sensor_noise)
                    from scale_consistency.model import simulate_observed_discrepancies

                    ratio_path = simulate_ratio_tracking(
                        RatioControlConfig(
                            a=0.5,
                            H=latent.H,
                            C_K=1.0,
                            C_S=1.0,
                            zeta0=latent.zeta,
                            ramp=0.0,
                            steps=max(4, t + 1),
                            noise_sigma=sensor_noise,
                            n0=16.0,
                            n_min=2.0,
                            n_max=4096.0,
                            decay=0.85,
                            gain=0.75,
                            deadband=0.10,
                        ),
                        policy="instant",
                        rng_seed=rng_seed + t,
                    )
                    discrepancy = np.mean(
                        [
                            simulate_observed_discrepancies(
                                lags,
                                zeta=latent.zeta,
                                H=latent.H,
                                sigma0=config.noise_sigma,
                                n=config.n_obs,
                                noise="heteroskedastic_power"
                                if sensor_noise > 0.12
                                else "gaussian",
                                heteroskedastic_alpha=sensor_noise,
                                heteroskedastic_beta=1.5,
                                rng=np.random.default_rng(
                                    int(rng.integers(0, 2**32 - 1))
                                ),
                            )
                            for _ in range(config.lag_reps)
                        ],
                        axis=0,
                    )
                    full_proxy = _estimate_proxy(
                        lags=lags,
                        discrepancy=discrepancy,
                        rho_window=ratio_path.rho_obs_path[
                            max(0, t - config.lag_count + 1) : t + 1
                        ],
                        config=fit_config,
                    )
                    mid = max(4, config.lag_count // 2)
                    short_proxy = _estimate_proxy(
                        lags=lags[:mid],
                        discrepancy=discrepancy[:mid],
                        rho_window=ratio_path.rho_obs_path[max(0, t - mid + 1) : t + 1],
                        config=fit_config,
                    )
                    long_proxy = _estimate_proxy(
                        lags=lags[mid:],
                        discrepancy=discrepancy[mid:],
                        rho_window=ratio_path.rho_obs_path[
                            max(0, t - (config.lag_count - mid) + 1) : t + 1
                        ],
                        config=fit_config,
                    )

                    chosen = (
                        _route_policy(
                            full_proxy,
                            type(
                                "_P",
                                (),
                                {
                                    "instant_noise_threshold": config.instant_noise_threshold,
                                    "lag_noise_threshold": config.lag_noise_threshold,
                                    "high_H_threshold": config.high_H_threshold,
                                },
                            )(),
                        )
                        if sensor_mode == "single"
                        else _route_meta_sensing(
                            full_proxy=full_proxy,
                            short_proxy=short_proxy,
                            long_proxy=long_proxy,
                            config=config,
                        )
                    )
                    current_policy, streak = _route_with_persistence(
                        current_policy=current_policy,
                        candidate_policy=chosen,
                        streak=streak,
                        switch_persistence=config.switch_persistence,
                    )

                    if t >= config.switch_step:
                        total_after_switch += 1
                        if current_policy == latent.oracle_policy:
                            correct_after_switch += 1
                        pre_route_cost += _mismatch_cost(
                            current_policy, latent.oracle_policy
                        )
                        if (
                            route_time == config.steps - 1
                            and current_policy == config.post_regime.oracle_policy
                        ):
                            route_time = t
                            oracle_switch_time = t

                route_delays.append(float(max(0, route_time - config.switch_step)))
                pre_route_costs.append(float(pre_route_cost))
                switch_accuracies.append(
                    correct_after_switch / max(total_after_switch, 1)
                )
                route_lag_to_oracle.append(float(route_time - oracle_switch_time))

            rows.append(
                MetaSensingRow(
                    sensor_mode=sensor_mode,
                    sensor_noise=sensor_noise,
                    mean_route_delay=float(np.mean(route_delays)),
                    median_route_delay=float(np.median(route_delays)),
                    mean_pre_route_cost=float(np.mean(pre_route_costs)),
                    mean_switch_accuracy=float(np.mean(switch_accuracies)),
                    mean_route_lag_to_oracle=float(np.mean(route_lag_to_oracle)),
                )
            )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run meta-sensing benchmark.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--steps", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = MetaSensingConfig(steps=args.steps, switch_step=args.steps // 2)
    rows = run_meta_sensing_benchmark(config=config, rng_seed=0)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print(
        "sensor_mode\tsensor_noise\tmean_route_delay\tmean_pre_route_cost\tmean_switch_accuracy"
    )
    for row in rows:
        print(
            f"{row.sensor_mode}\t{row.sensor_noise:.3f}\t{row.mean_route_delay:.4f}\t{row.mean_pre_route_cost:.4f}\t{row.mean_switch_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
