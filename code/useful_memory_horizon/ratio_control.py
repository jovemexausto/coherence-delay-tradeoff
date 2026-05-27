from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json

import numpy as np

from .partition_ratio import (
    ratio_at_optimizer,
    validity_partition_ratio,
)


@dataclass(frozen=True, slots=True)
class RatioControlConfig:
    a: float = 0.5
    H: float = 0.75
    C_K: float = 1.0
    C_S: float = 1.0
    zeta0: float = 0.01
    ramp: float = 0.0
    steps: int = 256
    noise_sigma: float = 0.15
    n0: float = 16.0
    n_min: float = 2.0
    n_max: float = 4096.0
    decay: float = 0.85
    gain: float = 0.75
    deadband: float = 0.10


@dataclass(frozen=True, slots=True)
class RatioControlPath:
    policy: str
    n_path: np.ndarray
    n_star_path: np.ndarray
    zeta_path: np.ndarray
    rho_true_path: np.ndarray
    rho_obs_path: np.ndarray
    score_path: np.ndarray
    relative_error_path: np.ndarray
    mean_relative_error: float
    mean_abs_log_update: float
    final_relative_error: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _target_ratio(config: RatioControlConfig) -> float:
    return ratio_at_optimizer(config.a, config.H)


def _zeta_path(config: RatioControlConfig) -> np.ndarray:
    return np.asarray(
        [config.zeta0 + config.ramp * t for t in range(config.steps)], dtype=float
    )


def simulate_ratio_tracking(
    config: RatioControlConfig,
    *,
    policy: str = "persistent",
    rng_seed: int = 0,
) -> RatioControlPath:
    if config.steps < 4:
        raise ValueError("steps must be at least 4")
    rng = np.random.default_rng(rng_seed)
    target = _target_ratio(config)
    zeta_path = _zeta_path(config)
    n_path = np.zeros(config.steps, dtype=float)
    n_star_path = np.zeros(config.steps, dtype=float)
    rho_true_path = np.zeros(config.steps, dtype=float)
    rho_obs_path = np.zeros(config.steps, dtype=float)
    score_path = np.zeros(config.steps, dtype=float)
    relative_error_path = np.zeros(config.steps, dtype=float)
    n = float(config.n0)
    score = 0.0
    last_n = n
    log_updates: list[float] = []

    for t, zeta in enumerate(zeta_path):
        n_star = float(
            ((config.a * config.C_K) / (config.H * config.C_S * zeta))
            ** (1.0 / (config.a + config.H))
        )
        rho_true = validity_partition_ratio(
            n, C_K=config.C_K, a=config.a, C_S=config.C_S, zeta=zeta, H=config.H
        )
        rho_obs = float(rho_true * np.exp(rng.normal(0.0, config.noise_sigma)))

        if policy == "instant":
            n_next = _clamp(
                n * ((target / rho_obs) ** (1.0 / (config.a + config.H))),
                config.n_min,
                config.n_max,
            )
            score = np.log(max(rho_obs, 1.0e-12) / target)
        elif policy == "persistent":
            evidence = float(np.log(max(rho_obs, 1.0e-12) / target))
            score = config.decay * score + (1.0 - config.decay) * evidence
            if abs(score) <= config.deadband:
                n_next = n
            else:
                n_next = _clamp(
                    n * np.exp(-config.gain * score), config.n_min, config.n_max
                )
        elif policy == "hold":
            n_next = n
            score = 0.0
        else:
            raise ValueError(f"unsupported policy: {policy}")

        n_path[t] = n
        n_star_path[t] = n_star
        rho_true_path[t] = rho_true
        rho_obs_path[t] = rho_obs
        score_path[t] = score
        relative_error_path[t] = abs(n - n_star) / max(n_star, 1.0e-12)
        log_updates.append(abs(np.log(max(n_next, 1.0e-12) / max(last_n, 1.0e-12))))
        last_n = n_next
        n = n_next

    return RatioControlPath(
        policy=policy,
        n_path=n_path,
        n_star_path=n_star_path,
        zeta_path=zeta_path,
        rho_true_path=rho_true_path,
        rho_obs_path=rho_obs_path,
        score_path=score_path,
        relative_error_path=relative_error_path,
        mean_relative_error=float(np.mean(relative_error_path)),
        mean_abs_log_update=float(np.mean(log_updates)),
        final_relative_error=float(relative_error_path[-1]),
    )


def run_ratio_control_benchmark(
    *,
    config: RatioControlConfig = RatioControlConfig(),
    rng_seed: int = 0,
) -> dict[str, dict[str, float | str]]:
    rows = []
    for policy in ("instant", "persistent", "hold"):
        path = simulate_ratio_tracking(config, policy=policy, rng_seed=rng_seed)
        rows.append(
            {
                "policy": policy,
                "mean_relative_error": path.mean_relative_error,
                "mean_abs_log_update": path.mean_abs_log_update,
                "final_relative_error": path.final_relative_error,
            }
        )
    return {row["policy"]: row for row in rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ratio-control benchmark.")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--noise-sigma", type=float, default=0.15)
    parser.add_argument("--ramp", type=float, default=0.0)
    parser.add_argument("--zeta0", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = RatioControlConfig(
        steps=args.steps,
        noise_sigma=args.noise_sigma,
        ramp=args.ramp,
        zeta0=args.zeta0,
    )
    result = run_ratio_control_benchmark(config=config, rng_seed=0)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=float))
        return
    print("policy\tmean_relative_error\tmean_abs_log_update\tfinal_relative_error")
    for policy in ("instant", "persistent", "hold"):
        row = result[policy]
        print(
            f"{policy}\t{row['mean_relative_error']:.6g}\t{row['mean_abs_log_update']:.6g}\t{row['final_relative_error']:.6g}"
        )


if __name__ == "__main__":
    main()
