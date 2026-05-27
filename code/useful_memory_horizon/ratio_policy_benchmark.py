from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
class RatioPolicyBenchmarkConfig:
    H_values: tuple[float, ...] = (0.3, 0.5, 0.75, 0.9)
    zeta0_values: tuple[float, ...] = (0.01, 0.05)
    ramp_values: tuple[float, ...] = (2.0e-5, 5.0e-5)
    ratio_noise_values: tuple[float, ...] = (0.05, 0.15, 0.3)
    lag_noise_values: tuple[float, ...] = (0.05, 0.15, 0.3)
    steps: int = 256
    lag_count: int = 24
    lag_reps: int = 12
    sigma0: float = 0.5
    n_obs: int = 500
    n0: float = 16.0
    n_min: float = 2.0
    n_max: float = 4096.0
    a: float = 0.5
    C_K: float = 1.0
    C_S: float = 1.0
    persistent_decay: float = 0.85
    persistent_gain: float = 0.75
    persistent_deadband: float = 0.10


@dataclass(frozen=True, slots=True)
class RatioPolicyBenchmarkRow:
    policy: str
    H: float
    zeta0: float
    ramp: float
    ratio_noise: float
    lag_noise: float
    mean_relative_error: float
    std_relative_error: float
    mean_abs_log_update: float
    final_relative_error: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _run_lag_geometry_policy(
    *,
    config: RatioPolicyBenchmarkConfig,
    H: float,
    zeta0: float,
    ramp: float,
    lag_noise: float,
    rng_seed: int,
) -> RatioPolicyBenchmarkRow:
    _ensure_temporalbridge_path()
    _ensure_scale_consistency_path()
    from temporalbridge.core.fit import fit_horizon
    from scale_consistency.model import simulate_observed_discrepancies

    rng = np.random.default_rng(rng_seed)
    lags = np.arange(1, config.lag_count + 1, dtype=float)
    n = float(config.n0)
    rel_err: list[float] = []
    updates: list[float] = []
    zeta_path = np.asarray([zeta0 + ramp * t for t in range(config.steps)], dtype=float)
    for zeta in zeta_path:
        discrepancy = np.mean(
            [
                simulate_observed_discrepancies(
                    lags,
                    zeta=float(zeta),
                    H=H,
                    sigma0=config.sigma0,
                    n=config.n_obs,
                    noise="heteroskedastic_power" if lag_noise > 0.1 else "gaussian",
                    heteroskedastic_alpha=lag_noise,
                    heteroskedastic_beta=1.5,
                    rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
                )
                for _ in range(config.lag_reps)
            ],
            axis=0,
        )
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
        n_star = float(fit["n_star"])
        n_next = _clamp(n_star, config.n_min, config.n_max)
        rel_err.append(abs(n - n_star) / max(n_star, 1.0e-12))
        updates.append(abs(np.log(max(n_next, 1.0e-12) / max(n, 1.0e-12))))
        n = n_next
    return RatioPolicyBenchmarkRow(
        policy="lag_geometry",
        H=H,
        zeta0=zeta0,
        ramp=ramp,
        ratio_noise=0.0,
        lag_noise=lag_noise,
        mean_relative_error=float(np.mean(rel_err)),
        std_relative_error=float(np.std(rel_err)),
        mean_abs_log_update=float(np.mean(updates)),
        final_relative_error=float(rel_err[-1]),
    )


def _run_ratio_policy(
    *,
    config: RatioPolicyBenchmarkConfig,
    H: float,
    zeta0: float,
    ramp: float,
    ratio_noise: float,
    policy: str,
    rng_seed: int,
) -> RatioPolicyBenchmarkRow:
    ratio_config = RatioControlConfig(
        a=config.a,
        H=H,
        C_K=config.C_K,
        C_S=config.C_S,
        zeta0=zeta0,
        ramp=ramp,
        steps=config.steps,
        noise_sigma=ratio_noise,
        n0=config.n0,
        n_min=config.n_min,
        n_max=config.n_max,
        decay=config.persistent_decay,
        gain=config.persistent_gain,
        deadband=config.persistent_deadband,
    )
    path = simulate_ratio_tracking(ratio_config, policy=policy, rng_seed=rng_seed)
    return RatioPolicyBenchmarkRow(
        policy=f"{policy}_ratio",
        H=H,
        zeta0=zeta0,
        ramp=ramp,
        ratio_noise=ratio_noise,
        lag_noise=0.0,
        mean_relative_error=path.mean_relative_error,
        std_relative_error=float(np.std(path.relative_error_path)),
        mean_abs_log_update=path.mean_abs_log_update,
        final_relative_error=path.final_relative_error,
    )


def run_ratio_policy_benchmark(
    *,
    config: RatioPolicyBenchmarkConfig = RatioPolicyBenchmarkConfig(),
    rng_seed: int = 0,
) -> list[RatioPolicyBenchmarkRow]:
    rows: list[RatioPolicyBenchmarkRow] = []
    for H in config.H_values:
        for zeta0 in config.zeta0_values:
            for ramp in config.ramp_values:
                for ratio_noise in config.ratio_noise_values:
                    rows.append(
                        _run_ratio_policy(
                            config=config,
                            H=H,
                            zeta0=zeta0,
                            ramp=ramp,
                            ratio_noise=ratio_noise,
                            policy="instant",
                            rng_seed=rng_seed,
                        )
                    )
                    rows.append(
                        _run_ratio_policy(
                            config=config,
                            H=H,
                            zeta0=zeta0,
                            ramp=ramp,
                            ratio_noise=ratio_noise,
                            policy="persistent",
                            rng_seed=rng_seed,
                        )
                    )
                for lag_noise in config.lag_noise_values:
                    rows.append(
                        _run_lag_geometry_policy(
                            config=config,
                            H=H,
                            zeta0=zeta0,
                            ramp=ramp,
                            lag_noise=lag_noise,
                            rng_seed=rng_seed,
                        )
                    )
    return rows


def summarize_rows(rows: list[RatioPolicyBenchmarkRow]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    keys = sorted({(row.policy, row.H, row.zeta0, row.ramp) for row in rows})
    for policy, H, zeta0, ramp in keys:
        subset = [
            row
            for row in rows
            if row.policy == policy
            and row.H == H
            and row.zeta0 == zeta0
            and row.ramp == ramp
        ]
        summary.append(
            {
                "policy": policy,
                "H": H,
                "zeta0": zeta0,
                "ramp": ramp,
                "mean_relative_error": float(
                    np.mean([row.mean_relative_error for row in subset])
                ),
                "mean_abs_log_update": float(
                    np.mean([row.mean_abs_log_update for row in subset])
                ),
                "best_noise": float(
                    min(row.ratio_noise or row.lag_noise for row in subset)
                ),
            }
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ratio policy benchmark.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--steps", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = RatioPolicyBenchmarkConfig(steps=args.steps)
    rows = run_ratio_policy_benchmark(config=config, rng_seed=0)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print(
        "policy\tH\tzeta0\tramp\tnoise\tmean_rel_err\tmean_abs_log_update\tfinal_rel_err"
    )
    for row in rows[: min(len(rows), 24)]:
        noise = row.ratio_noise if row.ratio_noise > 0.0 else row.lag_noise
        print(
            f"{row.policy}\t{row.H:.2f}\t{row.zeta0:.3f}\t{row.ramp:.2e}\t{noise:.3f}\t"
            f"{row.mean_relative_error:.4f}\t{row.mean_abs_log_update:.4f}\t{row.final_relative_error:.4f}"
        )


if __name__ == "__main__":
    main()
