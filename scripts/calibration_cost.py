#!/usr/bin/env python3
"""Calibration-cost sensitivity for the useful-memory horizon.

This script evaluates how much tracking error is lost when the horizon is
selected using a misestimated calibration constant ``C_K``.
"""

from __future__ import annotations

import math


def e_min(c_k: float, zeta: float) -> float:
    return 1.5 * (c_k ** (2.0 / 3.0)) * (zeta ** (1.0 / 3.0))


def delta_e_ratio(r: float) -> float:
    """Return Delta E / E_min as a function of r = C_hat / C."""
    return r ** (-1.0 / 3.0) + 0.5 * r ** (2.0 / 3.0) - 1.5


def delta_e_exact(c_hat: float, c_true: float, zeta: float) -> float:
    return e_min(c_true, zeta) * delta_e_ratio(c_hat / c_true)


def delta_e_quadratic(c_hat: float, c_true: float, zeta: float) -> float:
    """Second-order local approximation around r=1."""
    r = c_hat / c_true
    return (1.0 / 9.0) * e_min(c_true, zeta) * (r - 1.0) ** 2


def main() -> int:
    c_true = 1.0
    zeta = 0.01
    rows = [0.5, 0.75, 0.9, 1.1, 1.5, 2.0]

    print("Calibration-cost sensitivity")
    print("r = C_hat / C_true,  E_min-normalized excess error")
    print(f"{'r':>6} {'exact':>12} {'quadratic':>12} {'rel_err%':>10}")
    print("-" * 46)
    for r in rows:
        exact = delta_e_ratio(r)
        quad = delta_e_quadratic(r * c_true, c_true, zeta) / e_min(c_true, zeta)
        rel_err = abs(exact - quad) / max(abs(exact), 1e-12) * 100.0
        print(f"{r:>6.2f} {exact:>12.6f} {quad:>12.6f} {rel_err:>10.2f}")

    print()
    print("Local expansion around r=1:")
    print("  Delta E = (1/9) * E_min * (r - 1)^2 + O((r - 1)^3)")
    print("  So calibration error is locally quadratic and symmetric to second order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
