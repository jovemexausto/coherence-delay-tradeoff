from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json

import numpy as np

from .policy_router import RegimeProxy, _route_policy


@dataclass(frozen=True, slots=True)
class RegimeCell:
    H: float
    zeta: float
    noise: float
    oracle_policy: str


@dataclass(frozen=True, slots=True)
class RegimeRouteDelayConfig:
    pre_regime: RegimeCell = RegimeCell(0.3, 0.01, 0.05, "instant_ratio")
    post_regime: RegimeCell = RegimeCell(0.9, 0.01, 0.05, "lag_geometry")
    steps: int = 128
    switch_step: int = 64
    trials: int = 64
    sensor_sigma_H: float = 0.05
    sensor_sigma_noise: float = 0.05
    sensor_sigma_zeta: float = 0.05
    switch_persistence: int = 4
    instant_noise_threshold: float = 0.22
    lag_noise_threshold: float = 0.18
    high_H_threshold: float = 0.48


@dataclass(frozen=True, slots=True)
class RegimeRouteDelayRow:
    sensor_noise: float
    mean_route_delay: float
    median_route_delay: float
    mean_pre_route_cost: float
    mean_switch_accuracy: float
    mean_route_lag_to_oracle: float
    mean_oracle_switch_delay: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _sensor_proxy(
    cell: RegimeCell, rng: np.random.Generator, sigma: float
) -> RegimeProxy:
    noise_hat = max(cell.noise + rng.normal(0.0, sigma), 0.0)
    return RegimeProxy(
        H_hat=max(cell.H + rng.normal(0.0, sigma), 0.05),
        zeta_hat=max(cell.zeta * np.exp(rng.normal(0.0, sigma)), 1.0e-12),
        noise_hat=noise_hat,
        identifiability=float(_clamp(1.0 / (1.0 + noise_hat), 0.0, 1.0)),
    )


def _mismatch_cost(chosen: str, oracle: str) -> float:
    if chosen == oracle:
        return 0.0
    if {chosen, oracle} == {"instant_ratio", "persistent_ratio"}:
        return 0.5
    if {chosen, oracle} == {"instant_ratio", "lag_geometry"}:
        return 1.25
    if {chosen, oracle} == {"persistent_ratio", "lag_geometry"}:
        return 1.0
    return 1.5


def _route_with_persistence(
    *,
    current_policy: str,
    candidate_policy: str,
    streak: int,
    switch_persistence: int,
) -> tuple[str, int]:
    if candidate_policy == current_policy:
        return current_policy, 0
    streak += 1
    if streak >= switch_persistence:
        return candidate_policy, 0
    return current_policy, streak


def run_regime_route_delay_benchmark(
    *,
    config: RegimeRouteDelayConfig = RegimeRouteDelayConfig(),
    sensor_noise_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
    rng_seed: int = 0,
) -> list[RegimeRouteDelayRow]:
    rows: list[RegimeRouteDelayRow] = []
    rng = np.random.default_rng(rng_seed)

    for sensor_noise in sensor_noise_levels:
        route_delays: list[float] = []
        pre_route_costs: list[float] = []
        switch_accuracies: list[float] = []
        route_lag_to_oracle: list[float] = []
        oracle_switch_delays: list[float] = []

        for _trial in range(config.trials):
            current_policy = config.pre_regime.oracle_policy
            current_oracle = config.pre_regime.oracle_policy
            streak = 0
            route_time = config.steps - 1
            oracle_switch_time = config.switch_step
            correct_after_switch = 0
            total_after_switch = 0
            pre_route_cost = 0.0

            for t in range(config.steps):
                latent = (
                    config.pre_regime if t < config.switch_step else config.post_regime
                )
                true_proxy = RegimeProxy(
                    H_hat=latent.H,
                    zeta_hat=latent.zeta,
                    noise_hat=latent.noise,
                    identifiability=float(1.0 / (1.0 + latent.noise)),
                )
                observed_proxy = _sensor_proxy(latent, rng, sensor_noise)
                oracle_policy = _route_policy(true_proxy, config)
                candidate_policy = _route_policy(observed_proxy, config)
                current_policy, streak = _route_with_persistence(
                    current_policy=current_policy,
                    candidate_policy=candidate_policy,
                    streak=streak,
                    switch_persistence=config.switch_persistence,
                )

                if t >= config.switch_step:
                    total_after_switch += 1
                    if current_policy == oracle_policy:
                        correct_after_switch += 1
                    pre_route_cost += _mismatch_cost(current_policy, oracle_policy)
                    if (
                        route_time == config.steps - 1
                        and current_policy == config.post_regime.oracle_policy
                    ):
                        route_time = t
                        current_oracle = oracle_policy
                        oracle_switch_time = t

            route_delay = max(0, route_time - config.switch_step)
            route_delays.append(float(route_delay))
            pre_route_costs.append(float(pre_route_cost))
            switch_accuracies.append(correct_after_switch / max(total_after_switch, 1))
            route_lag_to_oracle.append(float(route_time - oracle_switch_time))
            oracle_switch_delays.append(float(config.switch_step))

        rows.append(
            RegimeRouteDelayRow(
                sensor_noise=sensor_noise,
                mean_route_delay=float(np.mean(route_delays)),
                median_route_delay=float(np.median(route_delays)),
                mean_pre_route_cost=float(np.mean(pre_route_costs)),
                mean_switch_accuracy=float(np.mean(switch_accuracies)),
                mean_route_lag_to_oracle=float(np.mean(route_lag_to_oracle)),
                mean_oracle_switch_delay=float(np.mean(oracle_switch_delays)),
            )
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime-route delay benchmark.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = RegimeRouteDelayConfig(
        trials=args.trials, steps=args.steps, switch_step=args.steps // 2
    )
    rows = run_regime_route_delay_benchmark(config=config, rng_seed=0)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print(
        "sensor_noise\tmean_route_delay\tmedian_route_delay\tmean_pre_route_cost\tmean_switch_accuracy"
    )
    for row in rows:
        print(
            f"{row.sensor_noise:.3f}\t{row.mean_route_delay:.4f}\t{row.median_route_delay:.4f}\t{row.mean_pre_route_cost:.4f}\t{row.mean_switch_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
