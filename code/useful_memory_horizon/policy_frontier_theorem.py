from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .meta_sensing_benchmark import _route_meta_sensing, _sensor_proxy
from .policy_router import RegimeProxy, _route_policy
from .regime_route_delay import RegimeCell, _mismatch_cost, _route_with_persistence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
FIGURE_ROOT = ARTIFACT_ROOT / "figures" / "policy_frontier"
TABLE_ROOT = ARTIFACT_ROOT / "tables" / "policy_frontier"
for root in (FIGURE_ROOT, TABLE_ROOT):
    root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class PolicyFrontierConfig:
    pre_regime: RegimeCell = RegimeCell(0.3, 0.01, 0.05, "instant_ratio")
    post_regime: RegimeCell = RegimeCell(0.9, 0.01, 0.05, "lag_geometry")
    steps: int = 48
    switch_step: int = 24
    trials: int = 32
    lag_count: int = 40
    lag_reps: int = 8
    noise_sigma: float = 0.5
    n_obs: int = 500
    sensor_noise_levels: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    switch_persistence: int = 4
    instant_noise_threshold: float = 0.22
    lag_noise_threshold: float = 0.18
    high_H_threshold: float = 0.48


@dataclass(frozen=True, slots=True)
class PolicyFrontierRow:
    sensor_mode: str
    sensor_noise: float
    mean_route_delay: float
    mean_avoidable_delay: float
    mean_pre_route_cost: float
    mean_avoidable_cost: float
    mean_switch_accuracy: float
    mean_route_lag_to_oracle: float


def _sensor_proxy_for_mode(
    cell: RegimeCell,
    rng: np.random.Generator,
    sigma: float,
    mode: str,
) -> RegimeProxy:
    if mode == "oracle":
        return RegimeProxy(
            H_hat=cell.H,
            zeta_hat=cell.zeta,
            noise_hat=cell.noise,
            identifiability=float(1.0 / (1.0 + cell.noise)),
        )
    if mode == "single":
        return _sensor_proxy(cell, rng, sigma)
    if mode == "multiscale":
        base = _sensor_proxy(cell, rng, sigma)
        short = RegimeProxy(
            H_hat=max(base.H_hat - 0.06, 0.05),
            zeta_hat=base.zeta_hat,
            noise_hat=max(base.noise_hat * 1.15, 0.0),
            identifiability=base.identifiability,
        )
        long = RegimeProxy(
            H_hat=min(base.H_hat + 0.06, 1.5),
            zeta_hat=base.zeta_hat,
            noise_hat=max(base.noise_hat * 0.85, 0.0),
            identifiability=base.identifiability,
        )

        # Majority vote among three proxies, using the same policy frontier logic.
        class _Cfg:
            instant_noise_threshold = 0.22
            lag_noise_threshold = 0.18
            high_H_threshold = 0.48

        votes = [
            _route_policy(base, _Cfg()),
            _route_policy(short, _Cfg()),
            _route_policy(long, _Cfg()),
        ]
        counts = {policy: votes.count(policy) for policy in set(votes)}
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(top) == 1 or top[0][1] > top[1][1]:
            chosen = top[0][0]
        else:
            avg_noise = (base.noise_hat + short.noise_hat + long.noise_hat) / 3.0
            avg_h = (base.H_hat + short.H_hat + long.H_hat) / 3.0
            if avg_noise >= 0.22:
                chosen = "persistent_ratio"
            elif avg_h >= 0.48 and avg_noise <= 0.18:
                chosen = "lag_geometry"
            else:
                chosen = "instant_ratio"

        # encode the chosen mode back into a proxy-like object for routing
        if chosen == "persistent_ratio":
            return RegimeProxy(
                base.H_hat,
                base.zeta_hat,
                max(base.noise_hat, short.noise_hat, long.noise_hat),
                base.identifiability,
            )
        if chosen == "lag_geometry":
            return RegimeProxy(
                max(base.H_hat, long.H_hat),
                base.zeta_hat,
                min(base.noise_hat, long.noise_hat),
                base.identifiability,
            )
        return base
    raise ValueError(f"unsupported mode: {mode}")


def _route_policy_mode(proxy: RegimeProxy, mode: str) -> str:
    class _Cfg:
        instant_noise_threshold = 0.22
        lag_noise_threshold = 0.18
        high_H_threshold = 0.48

    if mode == "oracle":
        return _route_policy(proxy, _Cfg())
    if mode == "single":
        return _route_policy(proxy, _Cfg())
    if mode == "multiscale":
        return _route_policy(proxy, _Cfg())
    raise ValueError(f"unsupported mode: {mode}")


def run_policy_frontier_benchmark(
    *,
    config: PolicyFrontierConfig = PolicyFrontierConfig(),
    rng_seed: int = 0,
) -> list[PolicyFrontierRow]:
    rng = np.random.default_rng(rng_seed)
    rows: list[PolicyFrontierRow] = []

    for sensor_noise in config.sensor_noise_levels:
        mode_results: dict[str, list[float]] = {
            mode: [] for mode in ("oracle", "single", "multiscale")
        }
        mode_costs: dict[str, list[float]] = {
            mode: [] for mode in ("oracle", "single", "multiscale")
        }
        mode_switch_acc: dict[str, list[float]] = {
            mode: [] for mode in ("oracle", "single", "multiscale")
        }
        mode_route_lag: dict[str, list[float]] = {
            mode: [] for mode in ("oracle", "single", "multiscale")
        }

        for _trial in range(config.trials):
            current = {
                mode: config.pre_regime.oracle_policy
                for mode in ("oracle", "single", "multiscale")
            }
            streak = {mode: 0 for mode in ("oracle", "single", "multiscale")}
            route_time = {
                mode: config.steps - 1 for mode in ("oracle", "single", "multiscale")
            }
            switch_correct = {mode: 0 for mode in ("oracle", "single", "multiscale")}
            switch_total = {mode: 0 for mode in ("oracle", "single", "multiscale")}
            costs = {mode: 0.0 for mode in ("oracle", "single", "multiscale")}

            for t in range(config.steps):
                latent = (
                    config.pre_regime if t < config.switch_step else config.post_regime
                )
                oracle_proxy = _sensor_proxy_for_mode(
                    latent, rng, sensor_noise, "oracle"
                )
                single_proxy = _sensor_proxy_for_mode(
                    latent, rng, sensor_noise, "single"
                )
                multi_proxy = _sensor_proxy_for_mode(
                    latent, rng, sensor_noise, "multiscale"
                )
                proxies = {
                    "oracle": oracle_proxy,
                    "single": single_proxy,
                    "multiscale": multi_proxy,
                }
                for mode, proxy in proxies.items():
                    chosen = _route_policy_mode(proxy, mode)
                    current[mode], streak[mode] = _route_with_persistence(
                        current_policy=current[mode],
                        candidate_policy=chosen,
                        streak=streak[mode],
                        switch_persistence=config.switch_persistence,
                    )
                    if t >= config.switch_step:
                        switch_total[mode] += 1
                        if current[mode] == latent.oracle_policy:
                            switch_correct[mode] += 1
                        costs[mode] += _mismatch_cost(
                            current[mode], latent.oracle_policy
                        )
                        if (
                            route_time[mode] == config.steps - 1
                            and current[mode] == config.post_regime.oracle_policy
                        ):
                            route_time[mode] = t

            oracle_delay = max(0, route_time["oracle"] - config.switch_step)
            for mode in ("oracle", "single", "multiscale"):
                delay = max(0, route_time[mode] - config.switch_step)
                mode_results[mode].append(float(delay))
                mode_costs[mode].append(float(costs[mode]))
                mode_switch_acc[mode].append(
                    switch_correct[mode] / max(switch_total[mode], 1)
                )
                mode_route_lag[mode].append(
                    float(route_time[mode] - route_time["oracle"])
                )

        for mode in ("oracle", "single", "multiscale"):
            rows.append(
                PolicyFrontierRow(
                    sensor_mode=mode,
                    sensor_noise=sensor_noise,
                    mean_route_delay=float(np.mean(mode_results[mode])),
                    mean_avoidable_delay=float(
                        np.mean(
                            np.maximum(
                                np.asarray(mode_results[mode])
                                - np.asarray(mode_results["oracle"]),
                                0.0,
                            )
                        )
                    ),
                    mean_pre_route_cost=float(np.mean(mode_costs[mode])),
                    mean_avoidable_cost=float(
                        np.mean(
                            np.maximum(
                                np.asarray(mode_costs[mode])
                                - np.asarray(mode_costs["oracle"]),
                                0.0,
                            )
                        )
                    ),
                    mean_switch_accuracy=float(np.mean(mode_switch_acc[mode])),
                    mean_route_lag_to_oracle=float(np.mean(mode_route_lag[mode])),
                )
            )
    return rows


def build_policy_frontier_figure(
    *,
    config: PolicyFrontierConfig = PolicyFrontierConfig(),
    rng_seed: int = 0,
    output_path: Path | None = None,
) -> None:
    rows = run_policy_frontier_benchmark(config=config, rng_seed=rng_seed)
    df = pd.DataFrame([asdict(row) for row in rows])
    df.to_csv(TABLE_ROOT / "policy_frontier_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    palette = {"oracle": "#444444", "single": "#1f77b4", "multiscale": "#d62728"}
    linestyles = {"oracle": "--", "single": "-", "multiscale": "-"}

    for mode, sub in df.groupby("sensor_mode", sort=False):
        sub = sub.sort_values("sensor_noise")
        axes[0].plot(
            sub["sensor_noise"],
            sub["mean_route_delay"],
            marker="o",
            linewidth=2.0,
            linestyle=linestyles.get(mode, "-"),
            color=palette.get(mode, "#333333"),
            label=mode,
        )
        axes[1].plot(
            sub["sensor_noise"],
            sub["mean_avoidable_delay"],
            marker="o",
            linewidth=2.0,
            linestyle=linestyles.get(mode, "-"),
            color=palette.get(mode, "#333333"),
            label=mode,
        )

    axes[0].set_title("Route Delay")
    axes[0].set_xlabel("sensor noise")
    axes[0].set_ylabel("mean route delay")
    axes[0].legend(frameon=False)

    axes[1].set_title("Avoidable Delay")
    axes[1].set_xlabel("sensor noise")
    axes[1].set_ylabel("mean avoidable delay")
    axes[1].legend(frameon=False)

    fig.savefig(
        output_path or (FIGURE_ROOT / "fig_policy_frontier_theorem.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_ROOT / "fig_policy_frontier_theorem.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy frontier figure.")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--switch-step", type=int, default=24)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = PolicyFrontierConfig(
        steps=args.steps, switch_step=args.switch_step, trials=args.trials
    )
    rows = run_policy_frontier_benchmark(config=config, rng_seed=0)
    build_policy_frontier_figure(config=config, rng_seed=0)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print(pd.DataFrame([asdict(row) for row in rows]))


if __name__ == "__main__":
    main()
