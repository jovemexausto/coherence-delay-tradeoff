from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import build_manifest_row, export_rows_csv, spawn_rng, stable_run_id
from .sinkhorn import (
    _pairwise_squared_distances,
    _sinkhorn_kernel_from_cost,
    debiased_sinkhorn_divergence_weighted,
)


@dataclass(slots=True)
class EmbeddedSinkhornClosureConfig:
    pairs: tuple[tuple[int, int], ...] = ((8, 2), (12, 2))
    epsilons: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    self_sample_sizes: tuple[int, ...] = (24, 48, 96, 160)
    remainder_sample_sizes: tuple[int, ...] = (24, 48, 96, 160)
    influence_sample_sizes: tuple[int, ...] = (24, 48, 96, 160)
    remainder_seed_count: int = 16
    influence_seed_count: int = 12
    span: float = 0.25
    grid_radius: float = 0.125
    sinkhorn_max_iters: int = 300
    sinkhorn_tol: float = 1e-9
    master_seed: int = 20260524


@dataclass(slots=True)
class EmbeddedSinkhornClosureResult:
    self_coupling_rows: list[dict[str, str | float]]
    remainder_rows: list[dict[str, str | float]]
    influence_rows: list[dict[str, str | float]]
    quadratic_rows: list[dict[str, str | float]]
    summary_rows: list[dict[str, str | float | bool]]


def _base_grid(intrinsic_dim: int, radius: float) -> np.ndarray:
    if intrinsic_dim != 2:
        raise ValueError(
            "the calibrated closure model is implemented for intrinsic_dim=2"
        )
    return np.asarray(
        [
            (-radius, -radius),
            (-radius, radius),
            (radius, -radius),
            (radius, radius),
        ],
        dtype=float,
    )


def _embed(points: np.ndarray, ambient_dim: int) -> np.ndarray:
    out = np.zeros((points.shape[0], ambient_dim), dtype=float)
    out[:, : points.shape[1]] = points
    return out


def _row_shift(n: int, lag: int, span: float) -> float:
    if n <= 1:
        return 0.0
    return span * float(lag) / float(n - 1)


def _row_support(
    n: int,
    lag: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    grid_radius: float,
) -> np.ndarray:
    base = _base_grid(intrinsic_dim, grid_radius).copy()
    base[:, 0] += _row_shift(n, lag, span)
    return _embed(base, ambient_dim)


def _target_support_and_weights(
    n: int,
    ambient_dim: int,
    intrinsic_dim: int,
    span: float,
    grid_radius: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    row_supports = [
        _row_support(n, lag, ambient_dim, intrinsic_dim, span, grid_radius)
        for lag in range(n)
    ]
    support = np.vstack(row_supports)
    weights = np.full(support.shape[0], 1.0 / support.shape[0], dtype=float)
    return support, weights, row_supports


def _sample_triangular_window(
    row_supports: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    picks = [support[rng.integers(0, support.shape[0])] for support in row_supports]
    return np.asarray(picks, dtype=float)


def _sample_iid_mixture(
    support: np.ndarray,
    weights: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.choice(support.shape[0], size=n, p=weights)
    return support[indices]


def _self_coupling_stats(
    support: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    cost = _pairwise_squared_distances(support, support)
    kernel = _sinkhorn_kernel_from_cost(cost, epsilon)
    weighted_kernel = kernel * weights[None, :]
    row_sum = np.maximum(np.sum(weighted_kernel, axis=1), 1e-300)
    transition = weighted_kernel / row_sum[:, None]
    stationary = weights * row_sum
    stationary /= float(np.sum(stationary))
    similarity = (
        np.sqrt(stationary)[:, None] * transition / np.sqrt(stationary)[None, :]
    )
    similarity = 0.5 * (similarity + similarity.T)
    eigvals = np.linalg.eigvalsh(similarity)
    eigvals = np.sort(np.real(eigvals))
    centered_radius = float(max(0.0, eigvals[-2])) if eigvals.size >= 2 else 0.0
    squared_centered_radius = centered_radius**2
    inverse_norm_sq = float(1.0 / max(1e-12, 1.0 - squared_centered_radius))
    return {
        "centered_radius": centered_radius,
        "squared_centered_radius": squared_centered_radius,
        "inverse_norm_sq": inverse_norm_sq,
        "support_size": float(support.shape[0]),
    }


def _sinkhorn_cost_to_exact_target(
    sample: np.ndarray,
    target_support: np.ndarray,
    target_weights: np.ndarray,
    epsilon: float,
    *,
    max_iters: int,
    tol: float,
) -> float:
    sample_weights = np.full(sample.shape[0], 1.0 / sample.shape[0], dtype=float)
    result = debiased_sinkhorn_divergence_weighted(
        sample,
        target_support,
        sample_weights,
        target_weights,
        epsilon,
        max_iters=max_iters,
        tol=tol,
    )
    return max(float(result.cost), 0.0)


def _support_deviation_sup(
    sample: np.ndarray,
    target_support: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    support_map = {
        tuple(point.tolist()): idx for idx, point in enumerate(target_support)
    }
    empirical = np.zeros_like(target_weights)
    for point in sample:
        empirical[support_map[tuple(point.tolist())]] += 1.0 / sample.shape[0]
    return float(np.max(np.abs(empirical - target_weights)))


def _empirical_weight_vector(
    sample: np.ndarray,
    target_support: np.ndarray,
) -> np.ndarray:
    support_map = {
        tuple(point.tolist()): idx for idx, point in enumerate(target_support)
    }
    empirical = np.zeros(target_support.shape[0], dtype=float)
    for point in sample:
        empirical[support_map[tuple(point.tolist())]] += 1.0 / sample.shape[0]
    return empirical


def _estimate_log_slope(sample_sizes: tuple[int, ...], values: list[float]) -> float:
    x = np.log(np.asarray(sample_sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(values, dtype=float), 1e-18))
    return float(np.polyfit(x, y, 1)[0])


def run_embedded_sinkhorn_closure(
    config: EmbeddedSinkhornClosureConfig | None = None,
) -> EmbeddedSinkhornClosureResult:
    if config is None:
        config = EmbeddedSinkhornClosureConfig()

    self_rows: list[dict[str, str | float]] = []
    remainder_rows: list[dict[str, str | float]] = []
    influence_rows: list[dict[str, str | float]] = []
    quadratic_rows: list[dict[str, str | float]] = []
    summary_rows: list[dict[str, str | float | bool]] = []

    for ambient_dim, intrinsic_dim in config.pairs:
        pair_self_rows: list[dict[str, str | float]] = []
        pair_remainder_rows: list[dict[str, str | float]] = []
        pair_influence_rows: list[dict[str, str | float]] = []

        for epsilon in config.epsilons:
            for n in config.self_sample_sizes:
                target_support, target_weights, _ = _target_support_and_weights(
                    n, ambient_dim, intrinsic_dim, config.span, config.grid_radius
                )
                stats = _self_coupling_stats(target_support, target_weights, epsilon)
                row = {
                    "experiment": "self_coupling",
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "epsilon": epsilon,
                    "sample_size": n,
                    "support_size": int(stats["support_size"]),
                    "centered_radius": round(stats["centered_radius"], 8),
                    "squared_centered_radius": round(
                        stats["squared_centered_radius"], 8
                    ),
                    "inverse_norm_sq": round(stats["inverse_norm_sq"], 8),
                }
                self_rows.append(row)
                pair_self_rows.append(row)

            tri_curve: list[float] = []
            iid_curve: list[float] = []
            for n in config.remainder_sample_sizes:
                target_support, target_weights, row_supports = (
                    _target_support_and_weights(
                        n,
                        ambient_dim,
                        intrinsic_dim,
                        config.span,
                        config.grid_radius,
                    )
                )
                tri_values: list[float] = []
                iid_values: list[float] = []
                tri_quadratic_ratios: list[float] = []
                iid_quadratic_ratios: list[float] = []
                for seed in range(config.remainder_seed_count):
                    tri_rng = spawn_rng(
                        config.master_seed,
                        "grid-tri",
                        ambient_dim,
                        intrinsic_dim,
                        epsilon,
                        n,
                        seed,
                    )
                    iid_rng = spawn_rng(
                        config.master_seed,
                        "grid-iid",
                        ambient_dim,
                        intrinsic_dim,
                        epsilon,
                        n,
                        seed,
                    )
                    tri_sample = _sample_triangular_window(row_supports, tri_rng)
                    iid_sample = _sample_iid_mixture(
                        target_support, target_weights, n, iid_rng
                    )
                    tri_cost = _sinkhorn_cost_to_exact_target(
                        tri_sample,
                        target_support,
                        target_weights,
                        epsilon,
                        max_iters=config.sinkhorn_max_iters,
                        tol=config.sinkhorn_tol,
                    )
                    iid_cost = _sinkhorn_cost_to_exact_target(
                        iid_sample,
                        target_support,
                        target_weights,
                        epsilon,
                        max_iters=config.sinkhorn_max_iters,
                        tol=config.sinkhorn_tol,
                    )
                    tri_values.append(tri_cost)
                    iid_values.append(iid_cost)
                    tri_delta = (
                        _empirical_weight_vector(tri_sample, target_support)
                        - target_weights
                    )
                    iid_delta = (
                        _empirical_weight_vector(iid_sample, target_support)
                        - target_weights
                    )
                    tri_l2_sq = float(np.sum(tri_delta * tri_delta))
                    iid_l2_sq = float(np.sum(iid_delta * iid_delta))
                    tri_quadratic_ratios.append(tri_cost / max(tri_l2_sq, 1e-18))
                    iid_quadratic_ratios.append(iid_cost / max(iid_l2_sq, 1e-18))
                tri_mean = float(np.mean(tri_values))
                iid_mean = float(np.mean(iid_values))
                tri_curve.append(tri_mean)
                iid_curve.append(iid_mean)
                tri_ratio_mean = float(np.mean(tri_quadratic_ratios))
                iid_ratio_mean = float(np.mean(iid_quadratic_ratios))
                pair_remainder_rows.extend(
                    [
                        {
                            "experiment": "linearization_remainder",
                            "ambient_dim": ambient_dim,
                            "intrinsic_dim": intrinsic_dim,
                            "epsilon": epsilon,
                            "sample_size": n,
                            "sample_role": "triangular_window",
                            "mean_cost": round(tri_mean, 10),
                        },
                        {
                            "experiment": "linearization_remainder",
                            "ambient_dim": ambient_dim,
                            "intrinsic_dim": intrinsic_dim,
                            "epsilon": epsilon,
                            "sample_size": n,
                            "sample_role": "iid_mixture",
                            "mean_cost": round(iid_mean, 10),
                        },
                    ]
                )
                remainder_rows.extend(pair_remainder_rows[-2:])
                quadratic_rows.extend(
                    [
                        {
                            "experiment": "quadratic_proxy",
                            "ambient_dim": ambient_dim,
                            "intrinsic_dim": intrinsic_dim,
                            "epsilon": epsilon,
                            "sample_size": n,
                            "sample_role": "triangular_window",
                            "mean_cost_to_l2sq": round(tri_ratio_mean, 10),
                            "max_cost_to_l2sq": round(
                                float(np.max(tri_quadratic_ratios)), 10
                            ),
                        },
                        {
                            "experiment": "quadratic_proxy",
                            "ambient_dim": ambient_dim,
                            "intrinsic_dim": intrinsic_dim,
                            "epsilon": epsilon,
                            "sample_size": n,
                            "sample_role": "iid_mixture",
                            "mean_cost_to_l2sq": round(iid_ratio_mean, 10),
                            "max_cost_to_l2sq": round(
                                float(np.max(iid_quadratic_ratios)), 10
                            ),
                        },
                    ]
                )

            for sample_role, ratio_source in (
                ("triangular_window", "tri_quadratic_ratios"),
                ("iid_mixture", "iid_quadratic_ratios"),
            ):
                role_rows = [
                    row
                    for row in quadratic_rows
                    if int(row["ambient_dim"]) == ambient_dim
                    and int(row["intrinsic_dim"]) == intrinsic_dim
                    and float(row["epsilon"]) == epsilon
                    and str(row["sample_role"]) == sample_role
                ]
                summary_rows.append(
                    {
                        "experiment": "quadratic_proxy",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_role": sample_role,
                        "max_mean_cost_to_l2sq": round(
                            max(float(row["mean_cost_to_l2sq"]) for row in role_rows), 8
                        ),
                        "min_mean_cost_to_l2sq": round(
                            min(float(row["mean_cost_to_l2sq"]) for row in role_rows), 8
                        ),
                        "max_pointwise_cost_to_l2sq": round(
                            max(float(row["max_cost_to_l2sq"]) for row in role_rows), 8
                        ),
                    }
                )

            tri_slope = _estimate_log_slope(config.remainder_sample_sizes, tri_curve)
            iid_slope = _estimate_log_slope(config.remainder_sample_sizes, iid_curve)
            max_ratio = float(
                np.max(
                    np.divide(
                        np.maximum(tri_curve, 1e-18),
                        np.maximum(iid_curve, 1e-18),
                    )
                )
            )
            summary_rows.extend(
                [
                    {
                        "experiment": "linearization_remainder",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_role": "triangular_window",
                        "slope": round(tri_slope, 6),
                        "carrier_a": round(-tri_slope, 6),
                        "max_tri_iid_ratio": round(max_ratio, 8),
                    },
                    {
                        "experiment": "linearization_remainder",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_role": "iid_mixture",
                        "slope": round(iid_slope, 6),
                        "carrier_a": round(-iid_slope, 6),
                        "max_tri_iid_ratio": round(max_ratio, 8),
                    },
                ]
            )

            for sample_role in ("triangular_window", "iid_mixture"):
                raw_curve: list[float] = []
                rootn_curve: list[float] = []
                for n in config.influence_sample_sizes:
                    target_support, target_weights, row_supports = (
                        _target_support_and_weights(
                            n,
                            ambient_dim,
                            intrinsic_dim,
                            config.span,
                            config.grid_radius,
                        )
                    )
                    values: list[float] = []
                    for seed in range(config.influence_seed_count):
                        rng = spawn_rng(
                            config.master_seed,
                            "grid-influence",
                            sample_role,
                            ambient_dim,
                            intrinsic_dim,
                            epsilon,
                            n,
                            seed,
                        )
                        if sample_role == "triangular_window":
                            sample = _sample_triangular_window(row_supports, rng)
                        else:
                            sample = _sample_iid_mixture(
                                target_support, target_weights, n, rng
                            )
                        values.append(
                            _support_deviation_sup(
                                sample, target_support, target_weights
                            )
                        )
                    mean_raw = float(np.mean(values))
                    mean_rootn = float(np.sqrt(n) * mean_raw)
                    raw_curve.append(mean_raw)
                    rootn_curve.append(mean_rootn)
                    row = {
                        "experiment": "influence_proxy",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_size": n,
                        "sample_role": sample_role,
                        "mean_raw_sup": round(mean_raw, 10),
                        "mean_rootn_sup": round(mean_rootn, 10),
                    }
                    influence_rows.append(row)
                    pair_influence_rows.append(row)

                raw_slope = _estimate_log_slope(
                    config.influence_sample_sizes, raw_curve
                )
                summary_rows.append(
                    {
                        "experiment": "influence_proxy",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_role": sample_role,
                        "raw_slope": round(raw_slope, 6),
                        "max_rootn_sup": round(float(np.max(rootn_curve)), 8),
                        "min_rootn_sup": round(float(np.min(rootn_curve)), 8),
                        "rootn_ratio": round(
                            float(
                                np.max(rootn_curve) / max(1e-12, np.min(rootn_curve))
                            ),
                            8,
                        ),
                    }
                )

        summary_rows.extend(
            [
                {
                    "experiment": "self_coupling",
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "worst_squared_centered_radius": round(
                        max(
                            float(row["squared_centered_radius"])
                            for row in pair_self_rows
                        ),
                        8,
                    ),
                    "worst_inverse_norm_sq": round(
                        max(float(row["inverse_norm_sq"]) for row in pair_self_rows), 8
                    ),
                    "positive_gap": all(
                        float(row["squared_centered_radius"]) < 1.0
                        for row in pair_self_rows
                    ),
                }
            ]
        )

    return EmbeddedSinkhornClosureResult(
        self_coupling_rows=self_rows,
        remainder_rows=remainder_rows,
        influence_rows=influence_rows,
        quadratic_rows=quadratic_rows,
        summary_rows=summary_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated embedded Sinkhorn closure diagnostics."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_embedded_closure"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EmbeddedSinkhornClosureConfig()
    result = run_embedded_sinkhorn_closure(config)
    export_rows_csv(result.self_coupling_rows, args.csv_dir / "self_coupling.csv")
    export_rows_csv(result.remainder_rows, args.csv_dir / "linearization_remainder.csv")
    export_rows_csv(result.influence_rows, args.csv_dir / "influence_proxy.csv")
    export_rows_csv(result.quadratic_rows, args.csv_dir / "quadratic_proxy.csv")
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")
    manifest = build_manifest_row(
        "sinkhorn_embedded_closure",
        {
            "pairs": config.pairs,
            "epsilons": config.epsilons,
            "self_sample_sizes": config.self_sample_sizes,
            "remainder_sample_sizes": config.remainder_sample_sizes,
            "influence_sample_sizes": config.influence_sample_sizes,
            "span": config.span,
            "grid_radius": config.grid_radius,
            "support_model": "calibrated-2d-grid",
        },
        run_id=stable_run_id(
            {
                "pairs": config.pairs,
                "epsilons": config.epsilons,
                "self_sample_sizes": config.self_sample_sizes,
                "remainder_sample_sizes": config.remainder_sample_sizes,
                "influence_sample_sizes": config.influence_sample_sizes,
                "grid_radius": config.grid_radius,
                "support_model": "calibrated-2d-grid",
            }
        ),
        seed=config.master_seed,
        notes=(
            "Calibrated embedded Sinkhorn closure diagnostics on a discrete k=2 support grid: "
            "S^2 self-coupling stability, exact-target null remainder slopes, finite-support empirical-process proxy, "
            "and support-growth quadratic cost-to-l2 diagnostics."
        ),
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")


if __name__ == "__main__":
    main()
