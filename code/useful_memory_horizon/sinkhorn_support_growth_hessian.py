from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import build_manifest_row, export_rows_csv, stable_run_id
from .sinkhorn import debiased_sinkhorn_divergence_weighted
from .sinkhorn_embedded_closure import _target_support_and_weights


@dataclass(slots=True)
class SupportGrowthHessianConfig:
    pairs: tuple[tuple[int, int], ...] = ((8, 2), (12, 2))
    epsilons: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    sample_sizes: tuple[int, ...] = (24, 48, 96)
    span: float = 0.25
    grid_radius: float = 0.125
    step_fraction: float = 0.25
    sinkhorn_max_iters: int = 300
    sinkhorn_tol: float = 1e-9


@dataclass(slots=True)
class SupportGrowthHessianResult:
    curvature_rows: list[dict[str, str | float | int]]
    summary_rows: list[dict[str, str | float | int]]


def _block_contrast_basis() -> np.ndarray:
    return (
        np.asarray(
            [
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=float,
        )
        / 2.0
    )


def _local_basis_vectors(num_blocks: int) -> list[tuple[str, int, np.ndarray]]:
    basis = _block_contrast_basis()
    vectors: list[tuple[str, int, np.ndarray]] = []
    total_dim = 4 * num_blocks
    for block_index in range(num_blocks):
        start = 4 * block_index
        for basis_index, block_basis in enumerate(basis):
            vector = np.zeros(total_dim, dtype=float)
            vector[start : start + 4] = block_basis
            vectors.append(("local", 3 * block_index + basis_index, vector))
    return vectors


def _collective_basis_vectors(num_blocks: int) -> list[tuple[str, int, np.ndarray]]:
    basis = _block_contrast_basis()
    vectors: list[tuple[str, int, np.ndarray]] = []
    repeated_scale = 1.0 / np.sqrt(float(num_blocks))
    for basis_index, block_basis in enumerate(basis):
        vector = np.tile(block_basis, num_blocks) * repeated_scale
        vectors.append(("collective", basis_index, vector))
    return vectors


def _null_sinkhorn_cost(
    support: np.ndarray,
    weights: np.ndarray,
    target_weights: np.ndarray,
    epsilon: float,
    *,
    max_iters: int,
    tol: float,
) -> float:
    result = debiased_sinkhorn_divergence_weighted(
        support,
        support,
        weights,
        target_weights,
        epsilon,
        max_iters=max_iters,
        tol=tol,
    )
    return max(float(result.cost), 0.0)


def run_support_growth_hessian_probe(
    config: SupportGrowthHessianConfig | None = None,
) -> SupportGrowthHessianResult:
    if config is None:
        config = SupportGrowthHessianConfig()

    curvature_rows: list[dict[str, str | float | int]] = []
    summary_rows: list[dict[str, str | float | int]] = []

    for ambient_dim, intrinsic_dim in config.pairs:
        for epsilon in config.epsilons:
            pair_rows: list[dict[str, str | float | int]] = []
            for n in config.sample_sizes:
                support, target_weights, _ = _target_support_and_weights(
                    n,
                    ambient_dim,
                    intrinsic_dim,
                    config.span,
                    config.grid_radius,
                )
                min_weight = float(np.min(target_weights))
                step = config.step_fraction * min_weight
                base_cost = _null_sinkhorn_cost(
                    support,
                    target_weights,
                    target_weights,
                    epsilon,
                    max_iters=config.sinkhorn_max_iters,
                    tol=config.sinkhorn_tol,
                )
                directions = _local_basis_vectors(n) + _collective_basis_vectors(n)
                for family, direction_index, direction in directions:
                    plus_weights = target_weights + step * direction
                    minus_weights = target_weights - step * direction
                    if (
                        float(np.min(plus_weights)) <= 0.0
                        or float(np.min(minus_weights)) <= 0.0
                    ):
                        raise ValueError(
                            "finite-difference step left the simplex interior"
                        )
                    plus_cost = _null_sinkhorn_cost(
                        support,
                        plus_weights,
                        target_weights,
                        epsilon,
                        max_iters=config.sinkhorn_max_iters,
                        tol=config.sinkhorn_tol,
                    )
                    minus_cost = _null_sinkhorn_cost(
                        support,
                        minus_weights,
                        target_weights,
                        epsilon,
                        max_iters=config.sinkhorn_max_iters,
                        tol=config.sinkhorn_tol,
                    )
                    curvature = max(
                        0.0,
                        (plus_cost - 2.0 * base_cost + minus_cost) / (step * step),
                    )
                    row = {
                        "experiment": "support_growth_hessian",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "sample_size": n,
                        "direction_family": family,
                        "direction_index": direction_index,
                        "step": round(step, 12),
                        "base_cost": round(base_cost, 12),
                        "curvature": round(curvature, 10),
                        "curvature_per_n": round(curvature / float(n), 10),
                    }
                    curvature_rows.append(row)
                    pair_rows.append(row)

            for family in ("local", "collective"):
                family_rows = [
                    row for row in pair_rows if str(row["direction_family"]) == family
                ]
                summary_rows.append(
                    {
                        "experiment": "support_growth_hessian",
                        "ambient_dim": ambient_dim,
                        "intrinsic_dim": intrinsic_dim,
                        "epsilon": epsilon,
                        "direction_family": family,
                        "max_curvature": round(
                            max(float(row["curvature"]) for row in family_rows), 8
                        ),
                        "min_curvature": round(
                            min(float(row["curvature"]) for row in family_rows), 8
                        ),
                        "max_curvature_per_n": round(
                            max(float(row["curvature_per_n"]) for row in family_rows), 8
                        ),
                        "min_curvature_per_n": round(
                            min(float(row["curvature_per_n"]) for row in family_rows), 8
                        ),
                        "mean_curvature": round(
                            float(
                                np.mean(
                                    [float(row["curvature"]) for row in family_rows]
                                )
                            ),
                            8,
                        ),
                    }
                )

            local_rows = [
                row for row in pair_rows if str(row["direction_family"]) == "local"
            ]
            trace_by_n = []
            for n in config.sample_sizes:
                rows_for_n = [row for row in local_rows if int(row["sample_size"]) == n]
                trace_n = float(np.sum([float(row["curvature"]) for row in rows_for_n]))
                trace_by_n.append((n, trace_n))
            trace_proxy = float(np.sum([trace_n for _, trace_n in trace_by_n]))
            trace_per_n_values = [trace_n / float(n) for n, trace_n in trace_by_n]
            cov_weighted_values = [
                trace_n / (4.0 * float(n) ** 2) for n, trace_n in trace_by_n
            ]
            summary_rows.append(
                {
                    "experiment": "support_growth_trace_proxy",
                    "ambient_dim": ambient_dim,
                    "intrinsic_dim": intrinsic_dim,
                    "epsilon": epsilon,
                    "trace_proxy": round(trace_proxy, 8),
                    "mean_trace_per_n": round(float(np.mean(trace_per_n_values)), 8),
                    "max_trace_per_n": round(float(np.max(trace_per_n_values)), 8),
                    "min_trace_per_n": round(float(np.min(trace_per_n_values)), 8),
                    "mean_cov_weighted_proxy": round(
                        float(np.mean(cov_weighted_values)), 10
                    ),
                    "max_cov_weighted_proxy": round(
                        float(np.max(cov_weighted_values)), 10
                    ),
                    "min_cov_weighted_proxy": round(
                        float(np.min(cov_weighted_values)), 10
                    ),
                }
            )

    return SupportGrowthHessianResult(
        curvature_rows=curvature_rows,
        summary_rows=summary_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe support-growth Hessian curvatures for null Sinkhorn."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/sinkhorn_support_growth_hessian"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SupportGrowthHessianConfig()
    result = run_support_growth_hessian_probe(config)
    export_rows_csv(result.curvature_rows, args.csv_dir / "curvature.csv")
    export_rows_csv(result.summary_rows, args.csv_dir / "summary.csv")
    manifest = build_manifest_row(
        "sinkhorn_support_growth_hessian",
        {
            "pairs": config.pairs,
            "epsilons": config.epsilons,
            "sample_sizes": config.sample_sizes,
            "span": config.span,
            "grid_radius": config.grid_radius,
            "step_fraction": config.step_fraction,
            "support_model": "calibrated-2d-support-growth-grid",
        },
        run_id=stable_run_id(
            {
                "pairs": config.pairs,
                "epsilons": config.epsilons,
                "sample_sizes": config.sample_sizes,
                "step_fraction": config.step_fraction,
                "support_model": "calibrated-2d-support-growth-grid",
            }
        ),
        notes=(
            "Numerical Hessian-curvature probe for the null Sinkhorn map on the calibrated "
            "support-growth grid."
        ),
    )
    export_rows_csv([manifest], args.csv_dir / "manifest.csv")


if __name__ == "__main__":
    main()
