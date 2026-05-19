from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import export_rows_csv
from .sinkhorn import debiased_sinkhorn_divergence
from .smooth_fixed_support_bahadur import (
    fixed_span_deformations,
    smooth_fixed_support_warp,
)


@dataclass(slots=True)
class SmoothSinkhornConfig:
    n_values: tuple[int, ...] = (40, 80, 160, 320)
    replications: int = 20
    span_values: tuple[float, ...] = (0.25,)
    H_values: tuple[float, ...] = (0.5, 1.0)
    epsilons: tuple[float, ...] = (0.50, 0.20, 0.10)
    max_iters: int = 250
    tol: float = 1e-8
    gap_tolerance: float = 0.15


@dataclass(slots=True)
class SmoothSinkhornResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def _sample_triangular(
    n: int, deformations: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    base = rng.uniform(0.0, 1.0, size=n)
    return smooth_fixed_support_warp(base, deformations)


def _sample_iid_mixture(
    n: int, deformations: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    base = rng.uniform(0.0, 1.0, size=n)
    component_index = rng.integers(0, len(deformations), size=n)
    return smooth_fixed_support_warp(base, deformations[component_index])


def _sinkhorn_cost(
    sample: np.ndarray,
    target: np.ndarray,
    epsilon: float,
    max_iters: int,
    tol: float,
) -> float:
    result = debiased_sinkhorn_divergence(
        sample[:, None],
        target[:, None],
        epsilon,
        max_iters=max_iters,
        tol=tol,
    )
    return max(float(result.cost), 0.0)


def run_smooth_fixed_support_sinkhorn(
    config: SmoothSinkhornConfig | None = None,
) -> SmoothSinkhornResult:
    if config is None:
        config = SmoothSinkhornConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    for span in config.span_values:
        for H in config.H_values:
            for epsilon in config.epsilons:
                tri_curve: list[float] = []
                iid_curve: list[float] = []
                for n in config.n_values:
                    deformations = fixed_span_deformations(n, span, H)
                    tri_values: list[float] = []
                    iid_values: list[float] = []
                    for rep in range(config.replications):
                        tri_rng = np.random.default_rng(
                            20_000_000
                            + 1000 * n
                            + 100 * rep
                            + int(100 * epsilon)
                            + int(1000 * H)
                        )
                        iid_rng = np.random.default_rng(
                            30_000_000
                            + 1000 * n
                            + 100 * rep
                            + int(100 * epsilon)
                            + int(1000 * H)
                        )
                        target_rng = np.random.default_rng(
                            40_000_000
                            + 1000 * n
                            + 100 * rep
                            + int(100 * epsilon)
                            + int(1000 * H)
                        )
                        tri_sample = _sample_triangular(n, deformations, tri_rng)
                        iid_sample = _sample_iid_mixture(n, deformations, iid_rng)
                        target_sample = _sample_iid_mixture(n, deformations, target_rng)
                        tri_values.append(
                            _sinkhorn_cost(
                                tri_sample,
                                target_sample,
                                epsilon,
                                config.max_iters,
                                config.tol,
                            )
                        )
                        iid_values.append(
                            _sinkhorn_cost(
                                iid_sample,
                                target_sample,
                                epsilon,
                                config.max_iters,
                                config.tol,
                            )
                        )

                    tri_mean = float(np.mean(tri_values))
                    iid_mean = float(np.mean(iid_values))
                    tri_curve.append(tri_mean)
                    iid_curve.append(iid_mean)
                    curve_rows.extend(
                        [
                            {
                                "experiment": "smooth-fixed-support-sinkhorn",
                                "setting": "triangular",
                                "span": round(span, 6),
                                "H": round(H, 6),
                                "epsilon": round(epsilon, 6),
                                "n": n,
                                "mean_cost": round(tri_mean, 10),
                            },
                            {
                                "experiment": "smooth-fixed-support-sinkhorn",
                                "setting": "iid-mixture",
                                "span": round(span, 6),
                                "H": round(H, 6),
                                "epsilon": round(epsilon, 6),
                                "n": n,
                                "mean_cost": round(iid_mean, 10),
                            },
                        ]
                    )

                log_n = np.log(np.asarray(config.n_values, dtype=float))
                tri_slope = float(
                    -np.polyfit(log_n, np.log(np.maximum(tri_curve, 1e-18)), 1)[0]
                )
                iid_slope = float(
                    -np.polyfit(log_n, np.log(np.maximum(iid_curve, 1e-18)), 1)[0]
                )
                gap = abs(tri_slope - iid_slope)
                status = (
                    "stable-core" if gap <= config.gap_tolerance else "noisy-boundary"
                )
                summary_rows.append(
                    {
                        "experiment": "smooth-fixed-support-sinkhorn",
                        "span": round(span, 6),
                        "H": round(H, 6),
                        "epsilon": round(epsilon, 6),
                        "tri_slope": round(tri_slope, 6),
                        "iid_slope": round(iid_slope, 6),
                        "slope_gap": round(gap, 6),
                        "status": status,
                    }
                )

    return SmoothSinkhornResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run smooth fixed-support Sinkhorn diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/smooth_fixed_support_sinkhorn"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smooth_fixed_support_sinkhorn()
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")
    export_rows_csv(result.curve_rows, args.csv_dir / "curves.csv")


if __name__ == "__main__":
    main()
