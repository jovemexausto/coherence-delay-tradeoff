from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json

import numpy as np


@dataclass(frozen=True, slots=True)
class PartitionRatioRow:
    n: float
    ratio: float
    ratio_over_a_over_H: float
    implied_n_star: float
    n_over_n_star: float
    side_of_horizon: str


def ratio_sweep_rows(
    *,
    n_grid: np.ndarray,
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
) -> list[PartitionRatioRow]:
    n_star = ((a * C_K) / (H * C_S * zeta)) ** (1.0 / (a + H))
    target = ratio_at_optimizer(a, H)
    rows: list[PartitionRatioRow] = []
    for n in n_grid:
        rho = validity_partition_ratio(n, C_K=C_K, a=a, C_S=C_S, zeta=zeta, H=H)
        implied = optimal_horizon_from_ratio(n, rho, a=a, H=H)
        rows.append(
            PartitionRatioRow(
                n=float(n),
                ratio=float(rho),
                ratio_over_a_over_H=float(rho / target),
                implied_n_star=float(implied),
                n_over_n_star=float(n / n_star),
                side_of_horizon="below"
                if n < n_star
                else "above"
                if n > n_star
                else "at",
            )
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect validity partition ratios.")
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--H", type=float, default=0.75)
    parser.add_argument("--C-K", dest="C_K", type=float, default=1.0)
    parser.add_argument("--C-S", dest="C_S", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=0.01)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--n-min", dest="n_min", type=float, default=2.0)
    parser.add_argument("--n-max", dest="n_max", type=float, default=128.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.count < 2:
        raise ValueError("count must be at least 2")
    n_grid = np.geomspace(args.n_min, args.n_max, args.count)
    rows = ratio_sweep_rows(
        n_grid=n_grid,
        C_K=args.C_K,
        a=args.a,
        C_S=args.C_S,
        zeta=args.zeta,
        H=args.H,
    )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return
    print("n\tratio\tratio/(a/H)\timplied_n_star\tn/n_star\tside")
    for row in rows:
        print(
            f"{row.n:.6g}\t{row.ratio:.6g}\t{row.ratio_over_a_over_H:.6g}\t"
            f"{row.implied_n_star:.6g}\t{row.n_over_n_star:.6g}\t{row.side_of_horizon}"
        )


def validity_partition_ratio(
    n: float,
    *,
    C_K: float,
    a: float,
    C_S: float,
    zeta: float,
    H: float,
) -> float:
    if n <= 0.0:
        raise ValueError("n must be positive")
    if C_K <= 0.0 or C_S <= 0.0 or zeta <= 0.0:
        raise ValueError("scale parameters must be positive")
    if a <= 0.0 or H <= 0.0:
        raise ValueError("exponents must be positive")
    return (C_S * zeta * n**H) / (C_K * n ** (-a))


def optimal_horizon_from_ratio(
    n: float,
    rho: float,
    *,
    a: float,
    H: float,
) -> float:
    if n <= 0.0 or rho <= 0.0:
        raise ValueError("n and rho must be positive")
    if a <= 0.0 or H <= 0.0:
        raise ValueError("a and H must be positive")
    return float(n * ((a / H) / rho) ** (1.0 / (a + H)))


def ratio_at_optimizer(a: float, H: float) -> float:
    if a <= 0.0 or H <= 0.0:
        raise ValueError("a and H must be positive")
    return a / H


if __name__ == "__main__":
    main()
