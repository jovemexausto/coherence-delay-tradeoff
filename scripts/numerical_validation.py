#!/usr/bin/env python3
"""Numerical validation suite for the paper's core propositions.

This is a lightweight reproducibility check, not a proof. It exercises three
claims:

1. Proposition 2.8 (Gaussian lower-bound construction);
2. Proposition 3.1 (EMA recovery-time formulas);
3. Theorem 2.1 (lag-inclusive upper bound for window averaging).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    name: str
    passed: bool
    details: str


def floor_theory(sigma: float, zeta: float) -> float:
    """Lower-bound floor used in the numerical check."""
    return (3.0 / 64.0) * sigma ** (2.0 / 3.0) * zeta ** (1.0 / 3.0)


def floor_sample(sigma: float, m: int) -> float:
    return sigma / (8.0 * np.sqrt(m))


def simulate_ramp_error(
    sigma: float,
    zeta: float,
    m: int,
    *,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Empirical error for the ramp construction used in the lower-bound check."""
    h_star = (sigma / zeta) ** (2.0 / 3.0)
    h = max(1, min(int(h_star), m))
    beta_h = min(zeta, sigma / (4.0 * h ** (3.0 / 2.0)))

    errors = []
    for _ in range(n_trials):
        sign = rng.choice([-1, 1])
        mu = np.array([sign * beta_h * max(h - j, 0) for j in range(m)], dtype=float)
        x = mu + rng.normal(0.0, sigma, size=m)
        mu_hat = float(np.mean(x[:h]))
        mu_true = float(mu[0])
        errors.append(abs(mu_hat - mu_true))

    return float(np.mean(errors))


def n_star_unclipped(delta: float, Ck: float) -> float:
    return (Ck / delta) ** (2.0 / 3.0)


def T_contract_theory(
    delta_slow: float, M: float, alpha: float, Ck: float, eps: float
) -> int:
    q = 1.0 - alpha
    nb = n_star_unclipped(M, Ck)
    delta_b = Ck * (nb + eps) ** (-3.0 / 2.0)
    arg = (M - delta_slow) / (M - delta_b)
    if arg <= 1.0:
        return 0
    return int(np.ceil(np.log(arg) / np.log(1.0 / q)))


def T_expand_theory(
    delta_slow: float, M: float, alpha: float, Ck: float, eps: float, T_burst: int
) -> int:
    q = 1.0 - alpha
    ns = n_star_unclipped(delta_slow, Ck)
    zeta_end = M - (M - delta_slow) * q**T_burst
    delta_s = Ck * (ns - eps) ** (-3.0 / 2.0)
    arg = (zeta_end - delta_slow) / (delta_s - delta_slow)
    if arg <= 1.0:
        return 0
    return int(np.ceil(np.log(arg) / np.log(1.0 / q)))


def T_contract_sim(
    delta_slow: float, M: float, alpha: float, Ck: float, eps: float
) -> int:
    nb = n_star_unclipped(M, Ck)
    d = delta_slow
    for k in range(10000):
        if abs(n_star_unclipped(d, Ck) - nb) <= eps:
            return k
        d = alpha * M + (1.0 - alpha) * d
    return 10000


def T_expand_sim(
    delta_slow: float, M: float, alpha: float, Ck: float, eps: float, T_burst: int
) -> int:
    d = delta_slow
    for _ in range(T_burst):
        d = alpha * M + (1.0 - alpha) * d

    ns = n_star_unclipped(delta_slow, Ck)
    for k in range(10000):
        if abs(n_star_unclipped(d, Ck) - ns) <= eps:
            return k
        d = alpha * delta_slow + (1.0 - alpha) * d
    return 10000


def sliding_window_error(
    zeta: float, n: int, sigma: float, *, T: int, n_seeds: int
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    tail_start = int(T * 0.6)
    errors = []

    for _ in range(n_seeds):
        mu = zeta * np.arange(T, dtype=float)
        x = mu + rng.normal(0.0, sigma, T)
        tail_errors = []
        for t in range(tail_start, T):
            window = x[max(0, t - n + 1) : t + 1]
            mu_hat = float(np.mean(window))
            tail_errors.append(abs(mu_hat - mu[t]))
        errors.append(float(np.mean(tail_errors)))

    return float(np.mean(errors)), float(np.std(errors))


def bound_theory(zeta: float, n: int, Ck: float) -> float:
    return Ck * n ** (-0.5) + 0.5 * zeta * n


def validate_prop_2_8(*, n_trials: int) -> Result:
    rng = np.random.default_rng(42)
    cases = [
        (1.0, 0.01, 100),
        (1.0, 0.005, 200),
        (1.0, 0.001, 1000),
        (0.5, 0.01, 50),
        (0.5, 0.005, 100),
        (2.0, 0.02, 200),
        (1.0, 0.05, 20),
    ]

    rows = []
    passed = True
    for sigma, zeta, m in cases:
        ft = floor_theory(sigma, zeta)
        fs = floor_sample(sigma, m)
        floor_max = max(ft, fs)
        err_emp = simulate_ramp_error(sigma, zeta, m, n_trials=n_trials, rng=rng)
        ok = err_emp >= 0.5 * floor_max
        passed = passed and ok
        rows.append(
            f"sigma={sigma:.2f}, zeta={zeta:.4f}, m={m}, floor={floor_max:.4f}, err={err_emp:.4f}, {'OK' if ok else 'FAIL'}"
        )

    return Result("Prop. 2.8", passed, " | ".join(rows))


def validate_prop_3_1() -> Result:
    alpha = 0.05
    Ck = 1.0
    eps_n = 3.0
    T_burst = 200
    cases = [
        (0.002, 0.004),
        (0.002, 0.006),
        (0.002, 0.008),
        (0.003, 0.006),
        (0.003, 0.009),
        (0.003, 0.012),
    ]

    rows = []
    passed = True
    for delta_slow, M in cases:
        ns = n_star_unclipped(delta_slow, Ck)
        nb = n_star_unclipped(M, Ck)
        if nb + eps_n >= ns:
            continue
        tc_th = T_contract_theory(delta_slow, M, alpha, Ck, eps_n)
        tc_sim = T_contract_sim(delta_slow, M, alpha, Ck, eps_n)
        te_th = T_expand_theory(delta_slow, M, alpha, Ck, eps_n, T_burst)
        te_sim = T_expand_sim(delta_slow, M, alpha, Ck, eps_n, T_burst)
        ok = abs(tc_th - tc_sim) <= 2 and abs(te_th - te_sim) <= 2
        passed = passed and ok
        rows.append(
            f"ds={delta_slow:.4f}, M={M:.4f}, Tc={tc_th}/{tc_sim}, Te={te_th}/{te_sim}, {'OK' if ok else 'FAIL'}"
        )

    return Result("Prop. 3.1", passed, " | ".join(rows))


def validate_theorem_2_1() -> Result:
    sigma = 1.0
    Ck = sigma
    cases = [
        (0.001, 10),
        (0.001, 50),
        (0.001, 100),
        (0.001, 200),
        (0.005, 10),
        (0.005, 30),
        (0.005, 50),
        (0.005, 100),
        (0.01, 5),
        (0.01, 20),
        (0.01, 50),
        (0.05, 5),
        (0.05, 10),
        (0.05, 20),
    ]

    rows = []
    passed = True
    for zeta, n in cases:
        err_mean, err_std = sliding_window_error(zeta, n, sigma, T=5000, n_seeds=30)
        bound = bound_theory(zeta, n, Ck)
        ok = bound >= err_mean
        passed = passed and ok
        rows.append(
            f"zeta={zeta:.4f}, n={n}, bound={bound:.4f}, err={err_mean:.4f}±{err_std:.4f}, {'OK' if ok else 'FAIL'}"
        )

    return Result("Thm. 2.1", passed, " | ".join(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=5000, help="Trials for the lower-bound check"
    )
    args = parser.parse_args()

    results = [
        validate_prop_2_8(n_trials=args.trials),
        validate_prop_3_1(),
        validate_theorem_2_1(),
    ]

    print("Numerical validation suite")
    print("=" * 80)
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}")
        print(result.details)
        print("-" * 80)

    ok = all(r.passed for r in results)
    print("Overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
