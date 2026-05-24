from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .common import export_rows_csv
from .finite_sample_geometry import exact_empirical_w2
from .sinkhorn import debiased_sinkhorn_divergence


DistanceFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(slots=True)
class CarrierRoughnessResearchConfig:
    raw_dims: tuple[int, ...] = (1, 2, 4, 6)
    ambient_intrinsic_pairs: tuple[tuple[int, int], ...] = ((4, 1), (8, 1), (8, 2))
    raw_sample_sizes: tuple[int, ...] = (16, 32, 64, 128)
    raw_seed_count: int = 10
    triangular_dims: tuple[int, ...] = (1, 2)
    H_values: tuple[float, ...] = (0.5, 1.0)
    fixed_spans: tuple[float, ...] = (0.1, 0.25, 0.5)
    span_growth_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    span_growth_base: float = 0.25
    fixed_zeta: float = 0.04
    triangular_sample_sizes: tuple[int, ...] = (32, 64, 128, 256)
    triangular_seed_count: int = 16
    sinkhorn_epsilons: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05)
    sinkhorn_ambient_intrinsic_pairs: tuple[tuple[int, int], ...] = ((8, 1),)
    sinkhorn_sample_sizes: tuple[int, ...] = (24, 48, 96, 160)
    sinkhorn_seed_count: int = 10


@dataclass(slots=True)
class CarrierRoughnessResearchResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def joint_horizon_exponent(a: float, H: float) -> float:
    if a <= 0.0 or H <= 0.0:
        raise ValueError("a and H must be positive")
    return 1.0 / (a + H)


def joint_minimum_error_exponents(a: float, H: float) -> tuple[float, float]:
    if a <= 0.0 or H <= 0.0:
        raise ValueError("a and H must be positive")
    denominator = a + H
    return H / denominator, a / denominator


def estimate_log_slope(sample_sizes: np.ndarray, values: list[float]) -> float:
    if len(sample_sizes) != len(values):
        raise ValueError("sample_sizes and values must have the same length")
    if len(values) < 2:
        raise ValueError("at least two values are required to estimate a slope")
    return float(np.polyfit(np.log(sample_sizes), np.log(values), 1)[0])


def _embedded_cube_sample(
    n: int, ambient_dim: int, intrinsic_dim: int, rng: np.random.Generator
) -> np.ndarray:
    if intrinsic_dim > ambient_dim:
        raise ValueError("intrinsic_dim cannot exceed ambient_dim")
    sample = np.zeros((n, ambient_dim), dtype=float)
    sample[:, :intrinsic_dim] = rng.uniform(-1.0, 1.0, size=(n, intrinsic_dim))
    return sample


def _path_means(n: int, dim: int, zeta: float, H: float) -> np.ndarray:
    lags = np.arange(n, dtype=float)
    means = np.zeros((n, dim), dtype=float)
    means[:, 0] = -zeta * lags**H
    return means


def _triangular_window_sample(
    n: int, dim: int, zeta: float, H: float, rng: np.random.Generator
) -> np.ndarray:
    return _path_means(n, dim, zeta, H) + rng.normal(size=(n, dim))


def _mixture_sample(
    n: int, dim: int, zeta: float, H: float, rng: np.random.Generator
) -> np.ndarray:
    means = _path_means(n, dim, zeta, H)
    component_index = rng.integers(0, n, size=n)
    return means[component_index] + rng.normal(size=(n, dim))


def _sqrt_sinkhorn(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    result = debiased_sinkhorn_divergence(x, y, epsilon)
    return float(abs(result.cost) ** 0.5)


def _embedded_fixed_span_means(n: int, ambient_dim: int, span: float) -> np.ndarray:
    raw = np.arange(n, dtype=float)
    raw -= raw.min()
    scale = float(raw.max()) if raw.max() > 0.0 else 1.0
    means = np.zeros((n, ambient_dim), dtype=float)
    means[:, 0] = span * raw / scale
    return means


def _embedded_uniform_window_sample(
    n: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if intrinsic_dim > ambient_dim:
        raise ValueError("intrinsic_dim cannot exceed ambient_dim")
    means = _embedded_fixed_span_means(n, ambient_dim, span)
    sample = means.copy()
    sample[:, :intrinsic_dim] += rng.uniform(-1.0, 1.0, size=(n, intrinsic_dim))
    return sample


def _embedded_uniform_mixture_sample(
    n: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if intrinsic_dim > ambient_dim:
        raise ValueError("intrinsic_dim cannot exceed ambient_dim")
    means = _embedded_fixed_span_means(n, ambient_dim, span)
    component_index = rng.integers(0, n, size=n)
    sample = means[component_index].copy()
    sample[:, :intrinsic_dim] += rng.uniform(-1.0, 1.0, size=(n, intrinsic_dim))
    return sample


def _estimate_mean_distance(
    sample_sizes: tuple[int, ...],
    seed_count: int,
    distance_fn: DistanceFn,
    sample_pair_fn: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    sizes = np.asarray(sample_sizes, dtype=int)
    means: list[float] = []
    for n in sizes:
        values: list[float] = []
        for seed in range(seed_count):
            left, right = sample_pair_fn(n, seed)
            values.append(distance_fn(left, right))
        means.append(float(np.mean(values)))
    return sizes, means


def _append_curve_rows(
    curve_rows: list[dict[str, str | float]],
    *,
    experiment: str,
    setting: str,
    sample_sizes: np.ndarray,
    means: list[float],
    metadata: dict[str, str | float],
) -> None:
    for sample_size, mean_value in zip(sample_sizes, means, strict=True):
        row: dict[str, str | float] = {
            "experiment": experiment,
            "setting": setting,
            "sample_size": int(sample_size),
            "mean_distance": round(mean_value, 6),
        }
        row.update(metadata)
        curve_rows.append(row)


def run_carrier_roughness_research(
    config: CarrierRoughnessResearchConfig | None = None,
) -> CarrierRoughnessResearchResult:
    if config is None:
        config = CarrierRoughnessResearchConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    for dim in config.raw_dims:
        sizes, means = _estimate_mean_distance(
            config.raw_sample_sizes,
            config.raw_seed_count,
            exact_empirical_w2,
            lambda n, seed, dim=dim: (
                np.random.default_rng(1_000 * dim + 10 * n + seed).uniform(
                    -1.0, 1.0, size=(n, dim)
                ),
                np.random.default_rng(2_000 * dim + 10 * n + seed).uniform(
                    -1.0, 1.0, size=(n, dim)
                ),
            ),
        )
        slope = estimate_log_slope(sizes, means)
        summary_rows.append(
            {
                "experiment": "raw-iid",
                "setting": f"raw W2 iid ambient d={dim}",
                "estimated_slope": round(slope, 4),
                "carrier_a": round(-slope, 4),
                "comment": "ambient-dimension benchmark",
            }
        )
        _append_curve_rows(
            curve_rows,
            experiment="raw-iid",
            setting=f"raw W2 iid ambient d={dim}",
            sample_sizes=sizes,
            means=means,
            metadata={"ambient_dim": dim, "intrinsic_dim": dim},
        )

    for ambient_dim, intrinsic_dim in config.ambient_intrinsic_pairs:
        sizes, means = _estimate_mean_distance(
            config.raw_sample_sizes,
            config.raw_seed_count,
            exact_empirical_w2,
            lambda n, seed, ambient_dim=ambient_dim, intrinsic_dim=intrinsic_dim: (
                _embedded_cube_sample(
                    n,
                    ambient_dim,
                    intrinsic_dim,
                    np.random.default_rng(5_000 + 100 * ambient_dim + 10 * n + seed),
                ),
                _embedded_cube_sample(
                    n,
                    ambient_dim,
                    intrinsic_dim,
                    np.random.default_rng(6_000 + 100 * ambient_dim + 10 * n + seed),
                ),
            ),
        )
        slope = estimate_log_slope(sizes, means)
        summary_rows.append(
            {
                "experiment": "intrinsic-iid",
                "setting": f"raw W2 iid ambient d={ambient_dim}, intrinsic k={intrinsic_dim}",
                "estimated_slope": round(slope, 4),
                "carrier_a": round(-slope, 4),
                "comment": "embedded low-dimensional support",
            }
        )
        _append_curve_rows(
            curve_rows,
            experiment="intrinsic-iid",
            setting=f"raw W2 iid ambient d={ambient_dim}, intrinsic k={intrinsic_dim}",
            sample_sizes=sizes,
            means=means,
            metadata={"ambient_dim": ambient_dim, "intrinsic_dim": intrinsic_dim},
        )

    for dim in config.triangular_dims:
        for H in config.H_values:
            for span in config.fixed_spans:
                fixed_span_triangular_means: list[float] = []
                fixed_span_iid_means: list[float] = []
                sizes = np.asarray(config.triangular_sample_sizes, dtype=int)
                for n in sizes:
                    zeta = span / (n**H)
                    fixed_span_triangular_values: list[float] = []
                    fixed_span_iid_values: list[float] = []
                    for seed in range(config.triangular_seed_count):
                        triangular_rng = np.random.default_rng(
                            10_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                        )
                        mixture_rng = np.random.default_rng(
                            20_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                        )
                        comparison_rng = np.random.default_rng(
                            30_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                        )
                        fixed_span_triangular_values.append(
                            exact_empirical_w2(
                                _triangular_window_sample(
                                    n, dim, zeta, H, triangular_rng
                                ),
                                _mixture_sample(n, dim, zeta, H, mixture_rng),
                            )
                        )
                        fixed_span_iid_values.append(
                            exact_empirical_w2(
                                _mixture_sample(n, dim, zeta, H, comparison_rng),
                                _mixture_sample(n, dim, zeta, H, comparison_rng),
                            )
                        )
                    fixed_span_triangular_means.append(
                        float(np.mean(fixed_span_triangular_values))
                    )
                    fixed_span_iid_means.append(float(np.mean(fixed_span_iid_values)))
                iid_slope = estimate_log_slope(sizes, fixed_span_iid_means)
                triangular_slope = estimate_log_slope(
                    sizes, fixed_span_triangular_means
                )
                summary_rows.append(
                    {
                        "experiment": "triangular-fixed-span",
                        "setting": f"iid mixture fixed span d={dim}, H={H:.1f}, span={span:.2f}",
                        "estimated_slope": round(iid_slope, 4),
                        "carrier_a": round(-iid_slope, 4),
                        "comment": "mixture benchmark",
                    }
                )
                summary_rows.append(
                    {
                        "experiment": "triangular-fixed-span",
                        "setting": f"triangular fixed span d={dim}, H={H:.1f}, span={span:.2f}",
                        "estimated_slope": round(triangular_slope, 4),
                        "carrier_a": round(-triangular_slope, 4),
                        "comment": "inheritance check",
                    }
                )
                _append_curve_rows(
                    curve_rows,
                    experiment="triangular-fixed-span",
                    setting=f"iid mixture fixed span d={dim}, H={H:.1f}, span={span:.2f}",
                    sample_sizes=sizes,
                    means=fixed_span_iid_means,
                    metadata={
                        "ambient_dim": dim,
                        "H": H,
                        "span": span,
                        "regime": "fixed-span",
                    },
                )
                _append_curve_rows(
                    curve_rows,
                    experiment="triangular-fixed-span",
                    setting=f"triangular fixed span d={dim}, H={H:.1f}, span={span:.2f}",
                    sample_sizes=sizes,
                    means=fixed_span_triangular_means,
                    metadata={
                        "ambient_dim": dim,
                        "H": H,
                        "span": span,
                        "regime": "fixed-span",
                    },
                )

            growing_span_triangular_means: list[float] = []
            growing_span_iid_means: list[float] = []
            sizes = np.asarray(config.triangular_sample_sizes, dtype=int)
            for n in sizes:
                zeta = config.fixed_zeta
                span = zeta * (n**H)
                growing_span_triangular_values: list[float] = []
                growing_span_iid_values: list[float] = []
                for seed in range(config.triangular_seed_count):
                    triangular_rng = np.random.default_rng(
                        40_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                    )
                    mixture_rng = np.random.default_rng(
                        50_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                    )
                    comparison_rng = np.random.default_rng(
                        60_000 + 1_000 * dim + 100 * int(10 * H) + 10 * n + seed
                    )
                    growing_span_triangular_values.append(
                        exact_empirical_w2(
                            _triangular_window_sample(n, dim, zeta, H, triangular_rng),
                            _mixture_sample(n, dim, zeta, H, mixture_rng),
                        )
                    )
                    growing_span_iid_values.append(
                        exact_empirical_w2(
                            _mixture_sample(n, dim, zeta, H, comparison_rng),
                            _mixture_sample(n, dim, zeta, H, comparison_rng),
                        )
                    )
                growing_span_triangular_means.append(
                    float(np.mean(growing_span_triangular_values))
                )
                growing_span_iid_means.append(float(np.mean(growing_span_iid_values)))
            iid_slope = estimate_log_slope(sizes, growing_span_iid_means)
            triangular_slope = estimate_log_slope(sizes, growing_span_triangular_means)
            summary_rows.append(
                {
                    "experiment": "triangular-growing-span",
                    "setting": f"iid mixture fixed zeta d={dim}, H={H:.1f}, zeta={config.fixed_zeta:.2f}",
                    "estimated_slope": round(iid_slope, 4),
                    "carrier_a": round(-iid_slope, 4),
                    "comment": "mixture benchmark under growing span",
                }
            )
            summary_rows.append(
                {
                    "experiment": "triangular-growing-span",
                    "setting": f"triangular fixed zeta d={dim}, H={H:.1f}, zeta={config.fixed_zeta:.2f}",
                    "estimated_slope": round(triangular_slope, 4),
                    "carrier_a": round(-triangular_slope, 4),
                    "comment": "heterogeneity stress test",
                }
            )
            _append_curve_rows(
                curve_rows,
                experiment="triangular-growing-span",
                setting=f"iid mixture fixed zeta d={dim}, H={H:.1f}, zeta={config.fixed_zeta:.2f}",
                sample_sizes=sizes,
                means=growing_span_iid_means,
                metadata={
                    "ambient_dim": dim,
                    "H": H,
                    "zeta": config.fixed_zeta,
                    "regime": "fixed-zeta",
                },
            )

            for growth_fraction in config.span_growth_fractions:
                span_growth_triangular_means: list[float] = []
                span_growth_iid_means: list[float] = []
                sizes = np.asarray(config.triangular_sample_sizes, dtype=int)
                for n in sizes:
                    gamma = growth_fraction * H
                    span = config.span_growth_base * (n**gamma)
                    zeta = span / (n**H)
                    span_growth_triangular_values: list[float] = []
                    span_growth_iid_values: list[float] = []
                    for seed in range(config.triangular_seed_count):
                        triangular_rng = np.random.default_rng(
                            100_000
                            + 1_000 * dim
                            + 100 * int(10 * H)
                            + 10 * n
                            + seed
                            + int(100 * growth_fraction)
                        )
                        mixture_rng = np.random.default_rng(
                            110_000
                            + 1_000 * dim
                            + 100 * int(10 * H)
                            + 10 * n
                            + seed
                            + int(100 * growth_fraction)
                        )
                        comparison_rng = np.random.default_rng(
                            120_000
                            + 1_000 * dim
                            + 100 * int(10 * H)
                            + 10 * n
                            + seed
                            + int(100 * growth_fraction)
                        )
                        span_growth_triangular_values.append(
                            exact_empirical_w2(
                                _triangular_window_sample(
                                    n, dim, zeta, H, triangular_rng
                                ),
                                _mixture_sample(n, dim, zeta, H, mixture_rng),
                            )
                        )
                        span_growth_iid_values.append(
                            exact_empirical_w2(
                                _mixture_sample(n, dim, zeta, H, comparison_rng),
                                _mixture_sample(n, dim, zeta, H, comparison_rng),
                            )
                        )
                    span_growth_triangular_means.append(
                        float(np.mean(span_growth_triangular_values))
                    )
                    span_growth_iid_means.append(float(np.mean(span_growth_iid_values)))
                iid_slope = estimate_log_slope(sizes, span_growth_iid_means)
                triangular_slope = estimate_log_slope(
                    sizes, span_growth_triangular_means
                )
                summary_rows.append(
                    {
                        "experiment": "triangular-span-growth",
                        "setting": f"iid mixture span growth d={dim}, H={H:.1f}, frac={growth_fraction:.2f}",
                        "estimated_slope": round(iid_slope, 4),
                        "carrier_a": round(-iid_slope, 4),
                        "comment": "mixture benchmark under growing span",
                    }
                )
                summary_rows.append(
                    {
                        "experiment": "triangular-span-growth",
                        "setting": f"triangular span growth d={dim}, H={H:.1f}, frac={growth_fraction:.2f}",
                        "estimated_slope": round(triangular_slope, 4),
                        "carrier_a": round(-triangular_slope, 4),
                        "comment": "intermediate heterogeneity regime",
                    }
                )
                _append_curve_rows(
                    curve_rows,
                    experiment="triangular-span-growth",
                    setting=f"iid mixture span growth d={dim}, H={H:.1f}, frac={growth_fraction:.2f}",
                    sample_sizes=sizes,
                    means=span_growth_iid_means,
                    metadata={
                        "ambient_dim": dim,
                        "H": H,
                        "span_growth_fraction": growth_fraction,
                        "regime": "span-growth",
                    },
                )
                _append_curve_rows(
                    curve_rows,
                    experiment="triangular-span-growth",
                    setting=f"triangular span growth d={dim}, H={H:.1f}, frac={growth_fraction:.2f}",
                    sample_sizes=sizes,
                    means=span_growth_triangular_means,
                    metadata={
                        "ambient_dim": dim,
                        "H": H,
                        "span_growth_fraction": growth_fraction,
                        "regime": "span-growth",
                    },
                )
            _append_curve_rows(
                curve_rows,
                experiment="triangular-growing-span",
                setting=f"triangular fixed zeta d={dim}, H={H:.1f}, zeta={config.fixed_zeta:.2f}",
                sample_sizes=sizes,
                means=growing_span_triangular_means,
                metadata={
                    "ambient_dim": dim,
                    "H": H,
                    "zeta": config.fixed_zeta,
                    "regime": "fixed-zeta",
                },
            )

    for ambient_dim, intrinsic_dim in config.sinkhorn_ambient_intrinsic_pairs:
        for epsilon in config.sinkhorn_epsilons:
            sizes = np.asarray(config.sinkhorn_sample_sizes, dtype=int)
            iid_means: list[float] = []
            triangular_means: list[float] = []
            for n in sizes:
                span = 0.25
                iid_values: list[float] = []
                triangular_values: list[float] = []
                for seed in range(config.sinkhorn_seed_count):
                    mixture_rng = np.random.default_rng(
                        70_000
                        + 1_000 * ambient_dim
                        + 100 * intrinsic_dim
                        + int(100 * epsilon)
                        + 10 * n
                        + seed
                    )
                    comparison_rng = np.random.default_rng(
                        80_000
                        + 1_000 * ambient_dim
                        + 100 * intrinsic_dim
                        + int(100 * epsilon)
                        + 10 * n
                        + seed
                    )
                    triangular_rng = np.random.default_rng(
                        90_000
                        + 1_000 * ambient_dim
                        + 100 * intrinsic_dim
                        + int(100 * epsilon)
                        + 10 * n
                        + seed
                    )
                    iid_values.append(
                        _sqrt_sinkhorn(
                            _embedded_uniform_mixture_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                span,
                                comparison_rng,
                            ),
                            _embedded_uniform_mixture_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                span,
                                comparison_rng,
                            ),
                            epsilon,
                        )
                    )
                    triangular_values.append(
                        _sqrt_sinkhorn(
                            _embedded_uniform_window_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                span,
                                triangular_rng,
                            ),
                            _embedded_uniform_mixture_sample(
                                n,
                                ambient_dim,
                                intrinsic_dim,
                                span,
                                mixture_rng,
                            ),
                            epsilon,
                        )
                    )
                iid_means.append(float(np.mean(iid_values)))
                triangular_means.append(float(np.mean(triangular_values)))
            iid_slope = estimate_log_slope(sizes, iid_means)
            triangular_slope = estimate_log_slope(sizes, triangular_means)
            iid_setting = f"Sinkhorn iid mixture ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, eps={epsilon:.2f}"
            triangular_setting = f"Sinkhorn triangular ambient d={ambient_dim}, intrinsic k={intrinsic_dim}, eps={epsilon:.2f}"
            summary_rows.append(
                {
                    "experiment": "sinkhorn-fixed-span",
                    "setting": iid_setting,
                    "sample_role": "iid_mixture",
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "epsilon": epsilon,
                    "span": span,
                    "regime": "fixed-span",
                    "estimated_slope": round(iid_slope, 4),
                    "carrier_a": round(-iid_slope, 4),
                    "comment": "fixed-span operational benchmark",
                }
            )
            summary_rows.append(
                {
                    "experiment": "sinkhorn-fixed-span",
                    "setting": triangular_setting,
                    "sample_role": "triangular_window",
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "epsilon": epsilon,
                    "span": span,
                    "regime": "fixed-span",
                    "estimated_slope": round(triangular_slope, 4),
                    "carrier_a": round(-triangular_slope, 4),
                    "comment": "fixed-span operational inheritance check",
                }
            )
            _append_curve_rows(
                curve_rows,
                experiment="sinkhorn-fixed-span",
                setting=iid_setting,
                sample_sizes=sizes,
                means=iid_means,
                metadata={
                    "epsilon": epsilon,
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "regime": "fixed-span",
                },
            )
            _append_curve_rows(
                curve_rows,
                experiment="sinkhorn-fixed-span",
                setting=triangular_setting,
                sample_sizes=sizes,
                means=triangular_means,
                metadata={
                    "epsilon": epsilon,
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "regime": "fixed-span",
                },
            )

    return CarrierRoughnessResearchResult(
        summary_rows=summary_rows, curve_rows=curve_rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run carrier-roughness research sweeps."
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("artifacts/csv/carrier_roughness_research")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_carrier_roughness_research()
    export_rows_csv(result.summary_rows, args.csv_dir / "carrier_roughness_summary.csv")
    export_rows_csv(result.curve_rows, args.csv_dir / "carrier_roughness_curves.csv")


if __name__ == "__main__":
    main()
